"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

import h5py
from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_dataset


pumped = PumpedDataset(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1",
                       progress=True,
                       mask_dir = None,
                       cycles = [1,2,3,4,5,6],
                       ignored_labels = None
                       )
pumped.process(correct_dark=True,
               correct_laser=True,
               output_dir = r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1\analysis")
save_dataset(pumped, 
             r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1\hlh_analysis\exp13_firstresult.h5")



