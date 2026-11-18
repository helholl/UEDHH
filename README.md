# uedhhlib

- is a package to analyze electron diffraction datasets (recorded at the Schweppes group at the MPSD Hamburg)

The main class is PumpedDataset which is used to convert a multitude of diffraction images into a h5 file. Subsequently the unpredictable electron sparcs must be filtered by hand and a new h5 file must be created, ignoring the images during sparcs. All this is collected in "ued_workflow.py" (or "ued_workflow.ipynb") under userscripts.

When the final h5 file is created the analysis varies, depending on the sample:

- Polycrystalline sample:
    1. get .poni file by opening pyfai-calib2 and using the mean_long_pumpoff.npy 
    (wavelength: 4.175716074647679e-2 Angström)
    2. run azigral.py script
- data_picker.py is a tool to manually check for arcs in the overall intensity of all images and extract a list of files to ignore during loading



## Requirments

required packages are

- ```numpy```
- ```matplotlib```
- ```pillow```
- ```tqdm```
- ```h5py```
- ```scipy```
- ```lmfit```


## uedhhlb

How to use:
```python
from uedhhlib import Dataset

dset = Dataset("path/to/measurement/folder", *args)

# saves dataset as iris compatible h5 file
dset.save("processed.h5")
```
*I know this is a sparse explanation, but I don't have more time at the moment. Ask for code examples or check docstrings*


## data_picker
How to use
```shell
python uedhhlib.data_picker.py path/to/processed.h5
```

This opens a window showing the absolute intensity of all recorded images on a lab time equivalent axis. With "+" you can add a ROI and move it with the mouse to a region you want to ignore. After all ROIs are added, press "space" which will print a list in the command window containing all tuples with file identifiers which is readable by the DataSet class. Use this list as

```python
from uedhhlib import Dataset
ignored_files = [(1, 103,9, 2), (1, 103,9, 1), (1, 103,75, 3), ..., (12, 101,65, 2)]

dset = Dataset("path/to/measurement/folder", ignore=ignored_files)
```

## Raw Data Structure

### Delay time ordering

The log file of each cycle contains the absolute times corresponding to the stage positions. The delay times are in reverse order compared to the absolute times. Hence, larger stage position corresponds to longer absolute time means earlier delay times