#!/usr/bin/env python
# coding: utf-8

# UED-Workflow in Schweppes Group in Hamburg
# Code Author Helena Hollstein
# 
# Fill out the next cell, save and then run cells individually but in the correct order!

# In[ ]:


import sys
import pandas as pd
from pathlib import Path
from uedhhlib.datasets import PumpedDataset
from uedhhlib.io.writers import save_datasets_h5, save_mean_pumpoff_long
from uedhhlib.tools import DataPicker
from pyqtgraph.Qt import QtWidgets

meas_path = Path(r"Z:\Users\Emma\Zyla1\2026\2026_11\NT52\261117\Au\Meas3") # path until MeasX
material_string = "Au"
exp_nr = 16 # Number of UED experiments on this material
cycles = (1, 2) #either as list, int or tuple(then from first to last entry including first and last entry)


# In[ ]:

#Load the data as hdf5 file for the first time to get the raw dataset

pumped = PumpedDataset(meas_path,
                       cycles = cycles)
pumped.process(correct_dark=False,
               correct_laser=False)
save_datasets_h5(pumped, 
             meas_path / "hlh_analysis" /f"{material_string}_exp{exp_nr}_raw_test.h5")
print("Done.")


# In[ ]:


# Use the datapicker to exclude sparcs. When the window opens, add aread with "+" and at the end save by "ctrl"+"S" as json file.
# CAUTION: If you give the jscon file a different name than "ignored_labels.json", you must change the "ignored_jsonpath" name in the next cell.

raw_h5_fpath = meas_path / "hlh_analysis" / f"{material_string}_exp{exp_nr}_raw_test.h5"

df = pd.read_hdf(raw_h5_fpath, key="file_registry")

# Prüfen, ob bereits eine Qt-Anwendung läuft, sonst eine neue erstellen
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

win = DataPicker(df, default_dir=str(Path(raw_h5_fpath).parent))
win.show()

# Statt sys.exit() nutzen wir in Jupyter .exec(), um den Kernel nicht abzuschießen
app.exec()


# In[ ]:


#Load the data as hdf5 file for the second time to get the dataset without arcs and corrected for dark and laser background.
# You also get a mean unpumped image as .npy file (use this one for pyfai-calib2)
# if the unpumped data is of interest (e.g. to check how the electron count or crystal quality has changed over labtime), add a output_dir and unpumped_fname to process to save an h5 file for unpumped images)

pumped = PumpedDataset(meas_path,
                       cycles = cycles,
                       ignored_jsonpath = meas_path /"hlh_analysis"/"ignored_labels_test.json"
                       )
pumped.process(correct_dark=True,
               correct_laser=True) 
save_mean_pumpoff_long(pumped, meas_path/"hlh_analysis"/"mean_long_pumpoff_test.npy")
save_datasets_h5(pumped, 
             meas_path/"hlh_analysis"/f"{material_string}_exp{exp_nr}_woarcs_test.h5")


# Now one can continue with analysis of this pumped h5 dataset.
# 
# If polycrystalline sample:
# 1. get .poni file by opening pyfai-calib2 and using the mean_long_pumpoff.npy 
# (wavelength: 4.175716074647679e-2 Angström)
# 2. run azigral.py script 
# 
# If single crystalline sample:
# 1. try peak_fitter.py
