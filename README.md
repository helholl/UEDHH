# uedhhlib

- is a package to analyze electron diffraction datasets (recorded at the Schweppes group at the MPSD Hamburg)


## Installation

You can install `uedhhlib` either using **Conda** or standard **Python virtual environments (`venv`)**.

### Prerequisites
Make sure you have either [Miniforge/Anaconda](https://github.com/conda-forge/miniforge) or a standard [Python](https://www.python.org/) (>= 3.8) installation on your system.

---

### Option A: Using Conda (Recommended)

1. Clone the repository and navigate to the project root directory:
   ```bash
   git clone <repository-url>
   cd uedhhlib


## How to use

The main class is PumpedDataset which is used to convert a multitude of diffraction images into a h5 file. Subsequently the unpredictable electron sparcs must be filtered by hand and a new h5 file must be created, ignoring the images during sparcs. All this is collected in "ued_workflow.py" (or "ued_workflow.ipynb") in userscripts.

When the final h5 file is created the analysis varies, depending on the sample:

1. Polycrystalline sample:
    - get the center of the diffraction rings by using pyfai-calib2 and saving the .poni file. 
    (use the mean_long_pumpoff.npy svaed during the last cell of "ued_workflow.py" and a wavelength: 4.175716074647679e-2 Angström)
    - run azigral.py script (in userscripts)
2. Singlecrystalline sample:
    - try the peak_fitter.py (in userscripts), if peaks can't be fitted well
    - integrate over area around peak and plot this over delaytime (script coming for that)


## Raw Data Structure

### Delay time ordering

The log file of each cycle contains the absolute times corresponding to the stage positions. The delay times are in reverse order compared to the absolute times. Hence, larger stage position corresponds to longer absolute time means earlier delay times