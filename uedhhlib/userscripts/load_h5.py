"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

import h5py
from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5


pumped = PumpedDataset(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2",
                       progress=True,
                       mask_dir = None,
                       cycles = [1,2],
                       ignored_labels = None,
                       ignored_jsonpath = r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis\ignored_labels_2cyc.json"
                       )
pumped.process(correct_dark=True,
               correct_laser=True,
               output_dir = r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis",
               unpumped_fname = "hlh_unpumped_woarcs.h5")
save_datasets_h5(pumped, 
             r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis\exp14_firstresult_2cyc_woarcs.h5")



