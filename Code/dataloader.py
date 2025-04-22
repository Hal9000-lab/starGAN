"""Main module."""

import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from constants import (__DATASET_DENOISING_PATH__,
                        __PREPROC_DATA_DENOISING_PATH__)
from utils import scan_sessions

AUTOTUNE = tf.data.experimental.AUTOTUNE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

__dataloader_modality__ = ["get", "gen"]


class DataLoader:
    def __init__(
        self,
        mode,
        data_path,
        label_a=None,
        label_b=None,
        output_dir=None,
    ):
        if mode not in __dataloader_modality__:
            raise ValueError(
                f"{mode} modality not recognized. Choose between 'gen' or 'get'")

        self.mode = mode

        if mode == "gen":
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"{data_path} does not exist")
            if not data_path.endswith(".json"):
                raise ValueError("data path must be a json file.")

            self.data_path = data_path

            if output_dir is None:
                base_dirname = os.path.dirname(self.data_path)
                output_dirname = os.path.basename(
                    self.data_path).replace(".json", "_TF")
                self.output_dir = os.path.join(base_dirname,
                                               output_dirname)
            else:
                self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

            if label_a is None or label_b is None:
                raise ValueError("label_a or label_b is None.")

            self.label_a = label_a #chest
            self.label_b = label_b #finger

            self.label_a_paths, self.label_b_paths = self.read_filepaths()

            dataset_property = {
                "label_a": self.label_a,
                "label_b": self.label_b,
            } #JSON file containing the strings for the input and reference names

            output_dir_content = os.listdir(self.output_dir)
            if output_dir_content is not None:
                if "ds_property.json" not in output_dir_content:
                    write_property = True
                elif len(output_dir_content) == 1 and "ds_property.json" in output_dir_content:
                    write_property = True
                else:
                    write_property = False
            else:
                write_property = False

            if write_property:
                with open(os.path.join(self.output_dir, "ds_property.json"),
                          'w', encoding="utf-8") as property_file:
                    json.dump(dataset_property, property_file, indent=2)

        elif mode == "get":
            if not os.path.exists(data_path) or (not os.path.isdir(data_path)):
                raise FileNotFoundError(f"{data_path} does not exist")
            self.output_dir = data_path

            self.label_a_paths = []
            self.label_b_paths = []
            with open(os.path.join(self.output_dir, "ds_property.json"),
                      'r', encoding="utf-8") as property_file:
                dataset_property = json.load(property_file)
                self.label_a = dataset_property["label_a"]
                self.label_b = dataset_property["label_b"]

    def read_filepaths(self):
        """Reads the paths stored in the JSON file. It returns a list for each key in the dictionary, with the respective paths corresponding to each key 
        (e.g., 'chest' and 'finger').

        Returns:
            list, list: two list containing the paths of every files for both
                classes. The list is sorted alphabetically, this can be usefull
                when files are named with a progressive number inside a folder
                (e.g.: 001.xxx, 002.xxx, ..., 999.xxx)
        """
        with open(self.data_path, 'r', encoding="utf-8") as f:
            dataset_paths = json.load(f)

        paths_label_a = dataset_paths[self.label_a]
        paths_label_b = dataset_paths[self.label_b]

        if len(paths_label_a) != len(paths_label_b):
            raise ValueError(
                f"Dimension mismatch: {len(paths_label_a)} != {len(paths_label_b)}")

        return paths_label_a, paths_label_b

    def get_dataset(self,
                    batch_size=32):
        
        """
        This function created the Tensorflow dataset.
        Input:
            batch_size (int): the batch size to be used for the dataset
        Returns: 
            tf.Data.Dataset: TensorFlow dataset with associated input and reference signals

        """

        ds = tf.data.Dataset.zip((self.get_signals(filepaths=self.label_a_paths,
                                                   label=self.label_a),
                                  self.get_signals(filepaths=self.label_b_paths,
                                                   label=self.label_b)
                                  ))

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def get_signals(self,
                    filepaths,
                    label,
                    ):
        
        """Open files for one class (either 'chest' or 'finger') and store it inside cache.

        This function performs all the (usually) slow reading operations that
        is necessary to execute at least the first time. After the first
        execution information are saved inside some cache file inside the specified cache
        folder. This function detects if cache files are already
        present, and in that case it skips the definition of these files.
        Please take into account that cache files will be as big as your
        Dataset overall size. First execution may result in a considerably
        bigger amount of time.

        Args:
            filepaths(str): Path to the signals to be read.
            label (str): either 'chest' or 'finger'; used for naming purposes only.

        Returns:
            tf.Data.Dataset: Tensorflow dataset object containing signals of one
                classes converted in Tensor format.
        """

        cache_file = os.path.join(self.output_dir, f"{label}.cache")
        index_file = f"{cache_file}.index"

        ds = tf.data.Dataset.from_tensor_slices(filepaths)

        ds = ds.map(lambda path: tf.py_function(self.read_npy_file,
                                                [path],
                                                [tf.float32],
                                                ),
                    num_parallel_calls=AUTOTUNE)

        ds = ds.unbatch()

        ds = ds.map(lambda sgn: tf.squeeze(sgn),
                    num_parallel_calls=AUTOTUNE)

        ds = ds.map(lambda x: x-tf.math.reduce_mean(x, axis=0))
        ds = ds.map(lambda x: x/tf.math.reduce_std(x, axis=0))

        ds = ds.cache(cache_file)

        if not os.path.exists(index_file):
            self._populate_cache(ds, cache_file, len(filepaths))

        return ds

    def read_npy_file(self, path):
        """Open a signal file and convert it to a tensor.

        Args:
            path(tf.Tensor): Tensor containing the path to the file to be
                opened.

        Returns:
            tf.Tensor: Tensor containing the actual signal content.
        """

        data = np.load(path.numpy().decode("utf-8")).astype(np.float32)
        return tf.convert_to_tensor(data)

    def _populate_cache(self, ds, cache_file, num_tot):
        """ Chache dataset """
        print(f"\n\tCaching decoded files in {cache_file}...")
        i = 0
        for _ in ds:
            i += 1
            sys.stdout.write("\r\t")
            sys.stdout.write(f"{i}/{num_tot}")
            sys.stdout.flush()

        print(f"\n\tCached decoded files in {cache_file}.\n")


def generate_dataset(data_path,
                     label_a,
                     label_b,
                     output_dir=None,
                     ):
    
    """
    This function is called in the main function (below) to generate the complete cached dataset, 
    which will later be read and split into training, validation, and test sets in the `trainer.py` script.

    Inputs:
    - `data_path` (str): Path to the JSON file containing the paths of the data to be read.
    - `label_a` (str): Name of the model input (e.g., 'chest').
    - `label_b` (str): Name of the model reference/target (e.g., 'finger').
    - `output_dir` (str): Directory where the cached dataset will be stored.

    Returns:
    - None. The cached dataset is saved in the specified output directory.
"""

    data_loader = DataLoader(mode="gen",
                             data_path=data_path,
                             label_a=label_a,
                             label_b=label_b,
                             output_dir=output_dir,
                             )

    data_loader.get_dataset()

    return


def get_dataset(data_dir,
                percentages,
                batch_size,
                ):
    """
This function is called in the `trainer.ipynb` script and splits the previously cached dataset into training, validation, and test sets.

Inputs:
- `data_dir` (str): Path to the directory where the cached dataset is stored.
- `percentages` (list or tuple of three floats): Specifies the proportions of data to allocate to the training, validation, and test sets, respectively.
- `batch_size` (int): The batch size to be used during training.

Returns:
- TensorFlow datasets for training, validation, and testing. These are used for model training and evaluation.
"""


    if len(percentages) != 3:
        raise ValueError("Percentages has to be a list of 3 elements")

    if round((percentages[0] + percentages[1] + percentages[2]), 1) != 1.0:
        raise ValueError("Sum of percentages has to be 1")

    data_loader = DataLoader(mode="get",
                             data_path=data_dir,
                             )

    complete_ds = data_loader.get_dataset(batch_size=batch_size)
    complete_ds = complete_ds.unbatch()

    num_sample = 0
    for _ in complete_ds:
        num_sample += 1

    train_ends = int(num_sample * percentages[0])
    valid_begins = train_ends
    valid_ends = valid_begins + int(num_sample * percentages[1])
    train_ds = complete_ds.take(train_ends)
    train_ds = train_ds.batch(batch_size)
    complete_ds = data_loader.get_dataset(batch_size=batch_size)
    complete_ds = complete_ds.unbatch()
    valid_ds = complete_ds.take(valid_ends)
    valid_ds = valid_ds.skip(valid_begins)
    valid_ds = valid_ds.batch(batch_size)
    test_ds = complete_ds.skip(valid_ends)
    test_ds = test_ds.batch(batch_size)
    return train_ds, valid_ds, test_ds


def generate_dataset_json(data_path):

    """
    Creates a `.json` file containing the paths to preprocessed data files.
    The resulting JSON file is a dictionary with two keys: `"chest"` and `"finger"`. Each key maps to a list of file paths corresponding 
    to the respective preprocessed signals. All data referenced in this file have already been preprocessed and passed quality inspection. 
    The preprocessed signals are stored in `.npy`.

    Input:  
    - `preprocessed_folder` (str): Path to the main folder containing preprocessed data.  
    This folder should include one subfolder per subject. Each subject folder must contain two files: `chest.npy` and `finger.npy`.

    Functionality: 
    The function scans all subfolders, retrieves the paths to `chest.npy` and `finger.npy` files, and adds them to the respective entries in the JSON dictionary.

    Output:
    - No return value.  
    - The JSON file is saved in the specified output directory.
    """


    sessions = scan_sessions(data_path=data_path,
                             session_type="preproc")

    chest_list = []
    finger_list = []
    for session in sessions:
        chest_list.append(f"{session}_chest.npy")
        finger_list.append(f"{session}_finger.npy")

    ds_dict = {"chest": chest_list,
               "finger": finger_list}

    os.makedirs(__DATASET_DENOISING_PATH__, exist_ok=True)

    with open(f"{__DATASET_DENOISING_PATH__}.json", "w", encoding="utf-8") as f:
        json.dump(ds_dict, f, indent=2)


if __name__ == "__main__":

    """
    Generate the whole dataset and cache it for faster future access. This script should be executed only once to set up the dataset.

    """

    generate_dataset_json(__PREPROC_DATA_DENOISING_PATH__) # We generate a JSON file that stores the file paths of the preprocessed data


    try:
        shutil.rmtree(__DATASET_DENOISING_PATH__)
    except:
        pass

    os.makedirs(__DATASET_DENOISING_PATH__, exist_ok=True)

    generate_dataset(data_path=f"{__DATASET_DENOISING_PATH__}.json",
                     label_a="chest",
                     label_b="finger",
                     output_dir=__DATASET_DENOISING_PATH__,
                     ) #Generate the whole dataset and cache it for faster future access
