"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

import h5py
from uedhhlib.datasets import BaseDataset, PumpedDataset, UnpumpedDataset
from uedhhlib.io.writers import save_dataset


pumped = PumpedDataset("Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1",
                       mask = None,
                       correct_laser = False,
                       all_imgs = False,
                       progess = True,
                       cycles = [1,2,3,4,5,6],
                       ignore = None,
                       norm = False
                       )
save_dataset(pumped, 
             "Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1\hlh_analysis\exp13_firstresult.h5")



