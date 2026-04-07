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
    df.to_hdf(filename, key="labtime/file_registry", mode="w") #??Key??


    with h5py.File(filename, "a") as f:
        f.create_dataset("time_points", data=ds.delay_times)
        f.create_dataset("valid_mask", data=~ds.mask)
        proc_group = f.create_group("processed")
        proc_group.create_dataset("equilibrium", data=ds.pump_off)
        proc_group.create_dataset("intensity", data=np.moveaxis(ds.data, 0, -1))

        labtime_group = f.create_group("lab_time")
        labtime_group.create_dataset("intensity", data=ds.lab_time_intensities) #total image intensity for arc detection: ONLY PUMPED??
        labtime_group.create_dataset("loaded_files", data=ds.loaded_files) #list of loaded files (abbreviated by delaystep, cycle and frame): PUMPED ONLY?
        labtime_group.create_dataset("timestamps", data=np.array(ds.timestamps, dtype="S")) #list of timestamps of taken images PUMPED ONLY
        labtime_group.create_dataset("intensity_unpumped", data=ds.lab_time_intensities_unpumped) #total intensity of unpumped images for mainbeam and crystal quality check? must be changed in future so that peak position is chosen for quality and short images are included for main beam
        labtime_group.create_dataset("loaded_files_unpumped", data=ds.loaded_files_unpumped) #list of loaded files UNPUMPED ONLY (abbreviated by delaystep, cycle and frame)
        labtime_group.create_dataset("timestamps_unpumped", data=np.array(ds.timestamps_unpumped, dtype="S")) #list of timestamps of taken images UNPUMPED ONLY
        labtime_group.create_dataset("loaded_files_pumponly", data=ds.loaded_files_pumponly) #list of loaded files LASER ONLY (abbreviated by delaystep, cycle and frame)
        labtime_group.create_dataset("laser_onlys", data= ds.pump_onlys) #AVERAGE? image of ALL? laser only images?
        labtime_group.create_dataset("timestamps_pumponly", data=np.array(ds.timestamps_pumponly, dtype="S")) #list of timestamps of taken images LASER ONLY
