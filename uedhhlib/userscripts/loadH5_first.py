"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5


pumped = PumpedDataset(r"Z:\Users\Emma\Zyla1\2026_06\260619\NT49_Pos5_Perylene\Meas1",
                       progress=True,
                       mask_dir = None,
                       cycles = (1,22),
                       ignored_labels = None,
                       ignored_jsonpath = None
                       )
pumped.process(correct_dark=False,
               correct_laser=False,
               output_dir = None,
               unpumped_fname = None)
save_datasets_h5(pumped, 
             r"Z:\Users\Emma\Zyla1\2026_06\260619\NT49_Pos5_Perylene\Meas1\hlh_analysis\exp16_raw.h5")



