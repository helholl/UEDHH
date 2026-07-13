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

    save_file_registry(filename=filename, registry=ds.file_registry, basedir=ds.basedir)

    # # create DataFrame from registry and save as hdf5 file
    # df = pd.DataFrame(ds.file_registry)
    # df.to_hdf(filename, key="file_registry", mode="w") 
    
    with h5py.File(filename, "a") as f:
        f.create_dataset("time_points", data=ds.delay_times)
        f.create_dataset("valid_mask", data=~ds.mask)
        proc_group = f.create_group("processed")
        proc_group.create_dataset("equilibrium", data=ds.pumpoff_long)
        proc_group.create_dataset("intensity", data=np.moveaxis(ds.pumped_data, 0, -1))
        proc_group.create_dataset("difference", data=np.moveaxis(ds.difference_data, 0, -1))
        if hasattr(ds, 'valid_delays_mask'):
            proc_group.create_dataset("valid_delays", data=ds.valid_delays_mask)
        if hasattr(ds, 'valid_delays_mask_diff'):
            proc_group.create_dataset("valid_delays_diff", data=ds.valid_delays_mask_diff)
            
def save_file_registry(filename, registry, basedir):
    """
    Save file registry to HDF5 efficiently.
    
    Parameters
    ----------
    filename : str
        Output HDF5 file
    registry : list
        File registry (list of dicts)
    basedir : str, optional
        Base directory (saved as attribute to reconstruct filepaths)
    """
    df = pd.DataFrame(registry)

    # convert h5_index to nullable integer (instead of h5_index = [None, None, 0, 1, 2, None, 3, 4, ...]-->None → pd.NA)
    if "h5_index" in df.columns:
        df["h5_index"] = df["h5_index"].astype("int64")

    # make certain all other elements are saved correctly
    if "img_type" in df.columns:
        df["img_type"] = df["img_type"].astype("category")
    if "cycle" in df.columns:
        df["cycle"] = df["cycle"].astype("int64")
    if "stage_position" in df.columns:
        df["stage_position"] = df["stage_position"].astype("float64")
    if "delay_time" in df.columns:
        df["delay_time"] = df["delay_time"].astype("float64")
    if "frame" in df.columns:
        df["frame"] = df["frame"].astype("int64")
    if "filepath" in df.columns:
        df["filepath"] = df["filepath"].astype("string")
    if "total_intensity" in df.columns:
        df["total_intensity"] = df["total_intensity"].astype("float64")
    if "timestamp" in df.columns:
        df["timestamp"] = df['timestamp'].astype('int64')

    df.to_hdf(filename, key="file_registry", mode="w", format="table")


    #basedir as metadata to be able to reconstruct filepaths with rel filepaths
    with pd.HDFStore(filename, mode="a") as store:
        store.get_storer("file_registry").attrs.basedir = basedir
