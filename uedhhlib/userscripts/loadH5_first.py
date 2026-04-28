"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5


pumped = PumpedDataset(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1_wohtpx",
                       progress=True,
                       mask_dir = None,
                       cycles = (1,6),
                       ignored_labels = None,
                       ignored_jsonpath = None
                       )
pumped.process(correct_dark=False,
               correct_laser=False,
               output_dir = None,
               unpumped_fname = None)
save_datasets_h5(pumped, 
             r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1_wohtpx\hlh_analysis\exp13_raw.h5")



