"""
Here the functions are defined which are used to save datasets. For now we are intersted in hdf5 files as output, which can be opened with iris
"""

import numpy as np
from os import PathLike
from pathlib import Path
import pandas as pd
import h5py


def save_datasets_h5(ds, filename: PathLike):
    """
    saves uedhhlib dataset as h5 file, which can be read by iris.

    Parameters
    ----------
    ds : (datatype?)
        uedhhlib dataset object
    filename : PathLike
        filename and path in which the h5 file should be stored
    """

    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    # create DataFrame from registry and save as hdf5 file
    df = pd.DataFrame(ds.file_registry)
    df.to_hdf(filename, key="file_registry", mode="w") 
    
    with h5py.File(filename, "a") as f:
        f.create_dataset("time_points", data=ds.delay_times)
        f.create_dataset("valid_mask", data=~ds.mask)
        proc_group = f.create_group("processed")
        proc_group.create_dataset("equilibrium", data=ds.pumpoff_long)
        proc_group.create_dataset("intensity", data=np.moveaxis(ds.pumped_data, 0, -1))

