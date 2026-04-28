"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5
from pathlib import Path

measX_path = Path(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1_wohtpx")

pumped = PumpedDataset(measX_path,
                       progress=True,
                       mask_dir = None,
                       cycles = (1,6),
                       ignored_labels = None,
                       ignored_jsonpath = measX_path /"hlh_analysis"/"exp13_ignored_labels.json"
                       )
pumped.process(correct_dark=True,
               correct_laser=True,
               output_dir = measX_path/"hlh_analysis",
               unpumped_fname = "exp13_unpumped_woarcs_hpxrm.h5")
save_datasets_h5(pumped, 
             measX_path/"hlh_analysis"/"exp13_woarcs.h5")



