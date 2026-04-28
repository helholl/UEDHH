from uedhhlib.datasets import PumpedDataset
import matplotlib.pyplot as plt
import pandas as pd

pumped = PumpedDataset(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2",
                       progress=True,
                       mask_dir = None,
                       cycles = (1,2),
                       ignored_labels = None,
                       ignored_jsonpath = None
)
pumped._init_cycles()
pumped._make_ignored_files_list()
pumped._load_delay_times()
pumped._load_dark_bckgr()
pumped._load_laser_bckgr()
pumped._load_mask()

pumped._load_and_save_unpumped(correct_dark=False, output_dir=None, unpumped_fname=None) # load pumpoff images in each cycle
pumped._load_and_save_pumped(correct_laser=False)# load the pumped files, average over cycles 

pumped.registry = pd.DataFrame(pumped.file_registry)
pumped.registry = pumped.registry.sort_values("timestamp").reset_index(drop=True)
# fig, ax = plt.subplots()

# data = pumped.dark_bckgr

# im = ax.imshow(data, cmap="inferno", vmin=0, vmax=200)
# fig.colorbar(im, ax=ax)
# #cbar = fig.colorbar()

# plt.show()


