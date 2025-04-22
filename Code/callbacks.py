"""This script defines the following callbacks:
    - `plotter`: used to visually display sample outputs during training.
    - `checkpoint`: used to save and load model checkpoints.
"""

import os
from datetime import datetime
import models
import matplotlib.pyplot as plt
import tensorflow as tf

from IPython.display import clear_output

class PlotterCallbackDenoising(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self,
                 dataset,
                 save_path="imgs/",
                 num_plot=4,
                 freq=1,
                 clear_every=5,
                 labels=None):
        
        """
        Args:
            dataset (tf.data.Dataset): Dataset containing the signals to plot.
            save_path (str): Path where the images will be saved.
            num_plot (int): Number of signals to plot at once.
            freq (int): Plotting frequency, defined in terms of training epochs.
            clear_every (int): Number of plots after which the page should be cleared.
            labels (list of str): Labels to display on the plots (e.g., "original", "restored", "target").
        """
        
        self.num_plot = num_plot
        self.signals = dataset.unbatch()
        self.save_path = save_path
        self.freq = freq
        self.clear_every = clear_every

        if not (os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)

        if labels is None:
            self.labels = ["Input",
                           "Target"
                           "Processed",
                           ]
        elif not isinstance(labels, list) or len(labels) != 3:
            raise ValueError(
                "'labels' parameters must be declared as a 3-elements list of strings.")
        else:
            self.labels = labels

    def on_epoch_end(self, epoch):
        if epoch % self.clear_every == 0:
            clear_output()

        if epoch % self.freq == 0:

            for i, pair in enumerate(self.signals.take(self.num_plot)):
                x, y = pair

                prediction = self.model.gen_A(tf.expand_dims(x,
                                                             axis=0))[0].numpy()
                
                _, ax = plt.subplots(3, 1, figsize=(10, 7))
                ax = ax.ravel()

                ax[0].plot(x[:, 0], color="orange", label=self.labels[0])
                ax[0].plot(y[:, 0], color="red", label=self.labels[1])
                ax[0].plot(prediction[:, 0],
                           color="blue",
                           label=self.labels[2],
                           alpha=0.7)
                ax[0].set_ylabel("RED")
                ax[0].legend(loc="upper left", ncol=3)
                ax[0].set_ylim([-4, 4]) #plot each channel in a separate subplot

                ax[1].plot(x[:, 1], color="grey", label=self.labels[0])
                ax[1].plot(y[:, 1], color="black", label=self.labels[1])
                ax[1].plot(prediction[:, 1],
                           color="blue",
                           label=self.labels[2],
                           alpha=0.7)
                ax[1].set_ylabel("IR")
                ax[1].legend(loc="upper left", ncol=3)
                ax[1].set_ylim([-4, 4])

                ax[2].plot(x[:, 2], color="lime", label=self.labels[0])
                ax[2].plot(y[:, 2], color="green", label=self.labels[1])
                ax[2].plot(prediction[:, 2],
                           color="blue",
                           label=self.labels[2],
                           alpha=0.7)
                ax[2].set_ylabel("GREEN")
                ax[2].legend(loc="upper left", ncol=3)
                ax[2].set_ylim([-4, 4])

                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path,
                                         f"generated_ppg_{i}_{epoch+1:03d}.png")) #save images

                tf.print(f"--- Example {i+1} ---")
                plt.show()
                plt.close()

class __CheckpointManager__():

    """A callback that creates a custom checkpoint manager to save checkpoints and restore them if already present."""

    def __init__(self, filepath, model):

        """
        Args:
            filepath (str): Path to store or retrieve the checkpoint.
            model: The model instance to be saved or restored.
        """

        self.filepath = filepath
        if isinstance(model, models.cycleGAN):
            self.ckpt = tf.train.Checkpoint(gen_A=model.gen_A,
                                            gen_B=model.gen_B,
                                            disc_X=model.disc_X,
                                            disc_Y=model.disc_Y,
                                            gen_A_optimizer=model.gen_A_optimizer,
                                            gen_B_optimizer=model.gen_B_optimizer,
                                            disc_X_optimizer=model.disc_X_optimizer,
                                            disc_Y_optimizer=model.disc_Y_optimizer,
                                            )

        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory=filepath,
                                                       max_to_keep=1,
                                                       )
        
    def save(self, checkpoint_number=None):
        self.ckpt_manager.save(checkpoint_number=checkpoint_number)
        print("Checkpoint saved.")

    def restore(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(
                f"Latest checkpoint restored: {self.ckpt_manager.latest_checkpoint}")
        else:
            print(f"No checkpoints found in {self.filepath} folder.")

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__( self,
                ckpt_dir,
                model,
                monitor="val_loss_A",
                mode="min",
                ref_val=0,
                warmup=0
            ):

        if mode not in ["min", "max", "always"]:
            raise ValueError(
                "mode parameter can be only 'min', 'max' or 'always'.")
        self.ckpt_dir = ckpt_dir
        self.manager = __CheckpointManager__(filepath=self.ckpt_dir,
                                             model=model,
                                             )

        self.model = model
        self.monitor = monitor
        self.mode = mode
        self.ref_val = ref_val
        self.warmup = warmup

    def on_epoch_end(self, epoch, logs=None):
        curr_monitor = logs[self.monitor]
        if epoch >= self.warmup:

            if (self.mode == "max" and curr_monitor > self.ref_val) or \
                    (self.mode == "min" and curr_monitor < self.ref_val) or \
                    self.mode == "always":

                if self.mode == "always":
                    print("Saving models....")
                else:
                    print(
                        f"{self.monitor} improved from {self.ref_val:.4f} to {curr_monitor:.4f}. Saving models...")
                    
                if curr_monitor > 1.0:
                    ckpt_id = int(curr_monitor * 1e2)
                else:
                    ckpt_id = int(curr_monitor * 1e4)

                self.manager.save(checkpoint_number=ckpt_id)
                self.ref_val = curr_monitor
            else:
                print(
                    f"{self.monitor} not improved. Current value: {curr_monitor:.4f} vs Reference value: {self.ref_val:.4f}. Skip saving.")
        else:
            print(f"Epoch {epoch+1} is in warm up phase. Skip saving.")

        return

def setup_callbacks(model,
                    save_logs=True,
                    save_checkpoints=True,
                    save_plots_denoising=False,
                    logs_dir=None,
                    ckpts_dir=None,
                    plots_dir=None,
                    validation_data=None,
                    ckpt_monitor='val_loss_A',
                    ckpt_mode= 'min',
                    plots_num=1,
                    plots_freq=1,
                    clear_every=5,
                    plot_labels=None,
                    ckpt_warmup = 0,
                    ):
    
    curr_time = datetime.now()
    curr_time = curr_time.strftime("%Y-%m-%d_%H-%M")

    callbacks_list = []
    if save_logs:
        if logs_dir is None:
            raise ValueError(
                "logs_dir parameter is mandatory when save_logs is True.")

        log_dir = os.path.join(logs_dir, str(model.id), curr_time)

        print('Log dir: ', log_dir)

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     write_graph=False,
                                                     profile_batch=0)
        callbacks_list.append(tb_callback)

    if save_checkpoints:
        if ckpts_dir is None:
            raise ValueError(
                "ckpts_dir parameter is mandatory when save_checkpoints is True.")

        if ckpt_monitor is None and ckpt_mode in ["min", "max"]:
            raise ValueError(
                "ckpt_monitor parameter is mandatory when save_checkpoints is True.")

        if validation_data is None:
            raise ValueError(
                "validation_data parameter is mandatory when save_checkpoints is True.")

        ckpt_dir = os.path.join(ckpts_dir, model.id, curr_time)

        metrics_dict = {}

        for m in model.metrics:
            metrics_dict[f"{m.name}"] = m

        if ckpt_monitor.replace("val_", "") not in metrics_dict.keys():
            raise ValueError(
                f"{ckpt_monitor} is not a valid metric for the compiled model.")

        ref_val = 5 

        if ckpt_mode in ["min", "max"]:
            print(f"Reference value for {ckpt_monitor}: {ref_val:.3f}")

        ckpt_callback = CheckpointCallback(ckpt_dir=ckpt_dir,
                                           model=model,
                                           monitor=ckpt_monitor,
                                           mode=ckpt_mode,
                                           ref_val=ref_val,
                                           warmup=ckpt_warmup,
                                           )

        callbacks_list.append(ckpt_callback)

    if save_plots_denoising:
        if plots_dir is None:
            raise ValueError(
                "plots_dir parameter is mandatory when save_plots is True.")

        if validation_data is None:
            raise ValueError(
                "validation_data parameter is mandatory when save_plots is True.")

        plot_dir = os.path.join(plots_dir, model.id, curr_time)
        print('Plot dir: ', plot_dir)

        plotter_callback = PlotterCallbackDenoising(dataset=validation_data,
                                                    save_path=plot_dir,
                                                    num_plot=plots_num,
                                                    freq=plots_freq,
                                                    clear_every=clear_every,
                                                    labels=plot_labels,
                                                    )

        callbacks_list.append(plotter_callback)

    return callbacks_list
