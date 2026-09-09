"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5


pumped = PumpedDataset(r"Z:\Users\Emma\Zyla1\2026\2026_07\NT52\260731\Pos2_Au_Exp10\Meas2",
                       progress=True,
                       mask_dir = None,
                       cycles = (1,11),
                       ignored_labels = None,
                       ignored_jsonpath = None
                       )
pumped.process(correct_dark=False,
               correct_laser=False,
               output_dir = None,
               unpumped_fname = None)
save_datasets_h5(pumped, 
             r"Z:\Users\Emma\Zyla1\2026\2026_07\NT52\260731\Pos2_Au_Exp10\Meas2\hlh_analysis\exp10_raw.h5")



