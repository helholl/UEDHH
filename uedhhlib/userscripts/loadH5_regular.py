"""
This script takes folders of UED data and loads them into datasets and stores these in an h5 file.
"""

from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5
from pathlib import Path

measX_path = Path(r"Z:\Users\Emma\Zyla1\2026\2026_07\NT52\260731\Pos2_Au_Exp10\Meas2")

pumped = PumpedDataset(measX_path,
                       progress=True,
                       mask_dir = None,
                       cycles = (1,9),
                       ignored_labels = None,
                       ignored_jsonpath = measX_path /"hlh_analysis"/"ignored_labels.json"
                       )
pumped.process(correct_dark=True,
               correct_laser=True,
               output_dir = None, #measX_path/"hlh_analysis",
               unpumped_fname = None) #"exp6_unpumped_woarcs.h5")
save_datasets_h5(pumped, 
             measX_path/"hlh_analysis"/"exp10_cyc1to9_woarcs.h5")



