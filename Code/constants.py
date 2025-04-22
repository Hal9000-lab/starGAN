import os

"""
This script defines all constants
"""

__SAMPLE_FREQ__ = 400
__CHUNK_SEC__ = 5
__OVERLAP_SEC__ = 2
__CHUNK_SAMPLE__ = __CHUNK_SEC__*__SAMPLE_FREQ__

__ROOT_PATH__ = os.path.dirname(os.path.dirname(__file__))

__DATA_DENOISING_PATH__ = os.path.join(__ROOT_PATH__, "Data")
__PREPROC_DATA_DENOISING_PATH__ = os.path.join(__DATA_DENOISING_PATH__,
                                               "preprocessed")
__DATASET_DENOISING_PATH__ = os.path.join(__DATA_DENOISING_PATH__, "datasets",
                                          f"dataset_{__CHUNK_SEC__}s_{__OVERLAP_SEC__}o_split")
__RESULT_DIR__ = os.path.join(__ROOT_PATH__, "results")