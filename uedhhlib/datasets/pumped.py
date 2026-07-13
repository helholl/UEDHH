from typing import Union
import numpy as np
import pandas as pd
from os import PathLike, listdir
from os.path import join, exists
from pathlib import Path
from re import findall
from scipy.constants import speed_of_light
from tqdm import tqdm
from datetime import datetime
import json
import h5py


class PumpedDataset:
    # 1. class attributes (None for now)

    # 2. __init__
    def __init__(
        self,
        basedir: PathLike,
        progress: bool = True,
        mask_dir: PathLike = None,
        cycles: Union[int, list, tuple] = None, 
        ignored_labels: list = None,
        ignored_jsonpath: PathLike = None
    ):
        """
        Loads full UED dataset taken in the SchwEpp group at the MPSD for further analysis.
        save() method saves h5 file which can be opened in Iris.
        Parameters
        ----------
        basedir : PathLike
            base directory cotaining the "Cycle X" directories
        progress : bool, optional
            if True a progress bar shows what is happening
        mask_dir : PathLike, optional
            constant mask applied to all images of the delay scans. Mask should be directory to an array of bool values where False values mark used points. If 'None' all points of the images are used, by default None
        cycles : Union[int, list, tuple], optional
            options:
            -if int only this cycle is loaded (e.g. 1 -> [1])
            -if list then only these explicit cycles are loaded (e.g. [1,2,5,6] -> [1,2,5,6])
            -if tuple of integers then all cycles from first to second integer (including both) are loaded (e.g. (1,3) -> [1,2,3])
            -if None: all cycles in the basedir
        ignored_labels : list
            list of tuples of the form (cycle_number, img_type, stage_position as float, frame number). To ignore frame three of unpumped image in cycle 5 at stage position 105.4 mm use (5, "unpumped", 105.4, 3).
        ignored_jsonpath : PathLike
            path to json file created by datapicker (Ctrl+S) in which files to be ignored are stored in the format (cycle_number, img_type, stage_position as float, frame number, filepath)
            the unambiguous labels are extracted therefrom.
            Note: If both ignored_labels and ignored_jsonpath are given, both are added
        """

        self.basedir = Path(basedir)
        self.progress = progress
        self.mask_dir = mask_dir
        self.cycles = cycles
        self.ignored_labels = ignored_labels
        self.ignored_jsonpath = ignored_jsonpath

        self.mask = None # space for file loaded from self.mask_dir later on
        self.file_registry = [] # list of librarys to store all relevant information per image
        
        
        self.ignored_files = [] # list of file directories to be ignored (at the moment only pumped!!!)
        self.stage_positions = [] # list of delay stage positions in mm
        self.rel_pos_zero = None # largest delay stage position at which the smalles temporal delay occurs
        self.delay_times = [] # list of delay times gathered from self.stage_positions in ps
        #self.timestamps = [] # for which img type?? Do I need a separate list for every img type?
        #self.labtime_intensities = [] # list of total intensities of all image types in labtime order (for arc removal)
        self.dark_bckgrs = [] # do I want list or average as class variable?
        self.dark_bckgr = None
        self.laser_bckgrs = [] # do I want list or average as class variable? list good when 1 laserbckgr per delay pos
        self.pumpoff_long = None # later save average pumpoff image for long exposure time here
        self.pumpoff_short = None # later save average pumpoff image for short exposure time here
        self.missing_cycles_per_delay = None # list with one entry per delaystep, counting skipped/ignored pumped files per delaystep
        self.valid_delays_mask = None #to store bool array of valid images of each delayposition in pumped data 
        self.pumped_data = None # list of 2d arrays (pumped imgs), one per delaystep, averaged over cycles
        self.missing_cycles_per_delay_diff = None # list for difference_data with one entry per delaystep, counting skipped/ignored pumped files per delaystep
        self.valid_delays_mask_diff = None #to store bool array of valid images of each delayposition in difference_data
        self.difference_data = None # list of 2d difference images per delaystep: every pumped image is subtracted by corresponding pumpoff_long (right before or after) and only after that averaged per delaystep

    # 3. private help methods (used by __init__)

    # 4. public main methods
    def process(
            self,
            correct_dark: bool = True,
            correct_laser: bool = True,
            output_dir: PathLike = None,
            save_diff: bool = True,
            unpumped_fname: str = "hlh_unpumped.h5"
            ):
        """
        Run process necessary to correct, average and normalize UED .npy files.

        Parameters
        ----------
        correct_dark : bool, optional
            if True the unpumped images are corrected by an average dark background image
        correct_laser : bool, optional
            if True the pumped images are corrected by a laser background image (at the moment average, later different laserOnly image per delay position)
        output_dir : PathLike, optional
            if output directory is given, the unpumped images are saved there in a h5 file sorted by labtime
        save_diff : bool, optional
            if True, additional to the pumped images there will be averaged difference images with each pumped image being subtracted by the corresponding unpumped_long in labtime.    
        unpumped_fname: str
            file name of unpumped h5 file
        """

        self._init_cycles() # creates list of cycles to be loaded
        self._make_ignored_files_list() # from self.ignored_labels create list with names of files to be ignored when loading images 
        self._load_delay_times() # convert delay stage positions to pump-probe delay times (compared to smallest delay time)
        self._load_dark_bckgr() # load all dark background images within the to be loaded cycles and save an average dark_bckgr
        self._load_laser_bckgr() # load all laser background images within the to be loaded cycles and save an average laser_bckgr (later list of laser_bckgrs per delay pos)
        self._load_mask() # load mask or create mask with ones from dark_bckgr shape
        self._load_and_save_unpumped(correct_dark=correct_dark, output_dir=output_dir, unpumped_fname=unpumped_fname) # load pumpoff images in each cycle
        self._load_and_save_pumped(correct_laser=correct_laser, save_diff=save_diff, correct_dark=correct_dark)# load the pumped files, average over cycles 

        self.registry = pd.DataFrame(self.file_registry)
        self.registry = self.registry.sort_values("timestamp").reset_index(drop=True)

    def _init_cycles(self):
        """
        Create list of cycles to be loaded, independent from input type (int, list or tuple)
        """
        if isinstance(self.cycles, int):
            self.cycles = [self.cycles]
        elif isinstance(self.cycles, list):
            pass
        elif isinstance(self.cycles, tuple):
            if len(self.cycles) == 2 and isinstance(self.cycles[0], int) and isinstance(self.cycles[1], int):
                start = self.cycles[0]
                end = self.cycles[1] + 1
                self.cycles = list(range(start, end))
            else:
                raise Exception("Cycle tuple must contain exactly two integers")
        elif self.cycles is None:
            cycle_dirs = [d for d in self.basedir.iterdir() if d.is_dir() and d.name.startswith("Cycle ")]
            self.cycles = sorted([int(d.split(" ")[1]) for d in cycle_dirs])

    def _make_ignored_files_list(self):
        """
        makes a list of filenames to be ignored during loading from list of labels or from jsonfile
        """
        if self.ignored_labels is None and self.ignored_jsonpath is None:
            return #nothing given to be ignored
        
        if self.ignored_labels is not None:
            if self.progress:
                print("compile list of ignored files from label list")

            for ign in self.ignored_labels:
                if len(ign)!=4:
                    print(f"ignored label must have 4 entrys! label: {ign}")
                    continue
                if ign[1] not in ("pumped", "unpumped_long", "unpumped_short","dark_bckgr","laser_bckgr"):
                    print(f"img type must be 'pumped', 'unpumped_long', 'unpumped_short', 'dark_bckgr' or 'laser_bckgr'. label: {ign}")
                    continue
                if not isinstance(ign[0], int):
                    print(f"first entry must be the cycle number as integer! label: {ign}")
                    continue
                if not isinstance(ign[2], float):
                    print(f"third entry must be the stage position as float! label: {ign}")
                if not isinstance(ign[3], int):
                    print(f"fourth entry must be the frame number as integer! label: {ign}")
                
                    match ign[1]:
                        case "pumped":
                            fname_id = "ProbeOnPumpOn_"
                        case "unpumped_long":
                            fname_id = "ProbeOnPumpOff_"
                        case "unpumped_short":
                            fname_id = "ProbeOnPumpOff_short"
                        case "dark_bckgr":
                            fname_id = "ProbeOffPumpOff_"
                        case "laser_bckgr":
                            fname_id = "ProbeOffPumpOn_"

                    self.ignored_files.append(
                        join(
                            self.basedir/ f"Cycle {int(ign[0])}"),
                            f"z_{fname_id}{str(ign[2]).replace('.', ',')} mm_Frm{int(ign[3])}.npy",
                        )

        if self.ignored_jsonpath is not None:
            if self.progress:
                print("compile list of ignored files from json file")

            with open(self.ignored_jsonpath) as f:
                d = json.load(f)
                for entry in d:
                    if entry["filepath"] not in self.ignored_files:
                        self.ignored_files.append(entry["filepath"])
        if self.progress:
            print(f"number of files to be ignored: {len(self.ignored_files)}")

    def _load_delay_times(self):
        """
        reads logfile and calculates delay times for all delay stage positions
        """
        if self.progress:
            print("accessing delay times from logfile")
        logfile_path = self.basedir / "Cycle 1"/ "logfile.txt"
        if exists(logfile_path):
            with open(logfile_path, 'r') as logf:
                for line in logf:
                    if line.startswith("#"):
                        continue
                    pos = float(line.strip().split("\t")[1])
                    self.stage_positions.append(round(pos, 2))
            self.stage_positions = sorted(self.stage_positions)
            self.rel_pos_zero = max(self.stage_positions) #smallest delaytime is arbitrarily set to 0 ps
        
        self.delay_times = [
            self.delaytime_from_stageposition(p)
            for p in self.stage_positions
        ]          
    
    def _load_dark_bckgr(self):
        """
        loads the dark background data of all cycles and makes a mean dark background image from all of them. 
        The unpumped data is corrected by the dark background and the dark background is used for its shape, because so far every measurement has dark background images
        """
        if self.progress:
            print("loading dark background images")
            cycles = tqdm(self.cycles)
        else:
            cycles = self.cycles
        for cycle in cycles:
            _cycle_path = self.basedir/ f"Cycle {int(cycle)}"
            _dbckgr_flist = []
            for f in listdir(_cycle_path):
                rel_fpath = join(f"Cycle {cycle}", f)
                if "ProbeOffPumpOff" in f and f.endswith(".npy") and rel_fpath not in self.ignored_files:
                    _dbckgr_flist.append(join(_cycle_path, f))

            for fpath in sorted(_dbckgr_flist, key=lambda p: (
                int(Path(p).parts[-2].replace("Cycle ","")),
                float(Path(p).stem.split("_")[2].replace(",",".").replace(" mm","")))
                ):
                dbckgr_img = np.load(fpath).T
                self._add_to_registry(fpath, intensity=dbckgr_img.sum())
                self.dark_bckgrs.append(dbckgr_img)

        if self.dark_bckgrs:
            self.dark_bckgr = np.mean(np.array(self.dark_bckgrs), axis=0)
        else:
            self.dark_bckgr = None

    def _load_laser_bckgr(self):
        """
        loads the laser background data of all cycles and makes a mean laser background image from all of them. Soon want option to take laser background per delay stage position.
        The pumped data is corrected by the laser background.
        """
        if self.progress:
            print("loading laser background images")
            cycles = tqdm(self.cycles)
        else:
            cycles = self.cycles
        for cycle in cycles:
            _cycle_path = self.basedir/ f"Cycle {int(cycle)}"
            _lbckgr_flist = []
            for f in listdir(_cycle_path):
                rel_fpath = join(f"Cycle {cycle}", f)
                if "ProbeOffPumpOn" in f and f.endswith(".npy") and rel_fpath not in self.ignored_files:
                    _lbckgr_flist.append(join(_cycle_path, f))

            for fpath in sorted(_lbckgr_flist, key=lambda p: (
                int(Path(p).parts[-2].replace("Cycle ","")),
                float(Path(p).stem.split("_")[2].replace(",",".").replace(" mm","")))       
                ):
                lbckgr_img = np.load(fpath).T
                self._add_to_registry(fpath, intensity=lbckgr_img.sum())

                self.laser_bckgrs.append(lbckgr_img)

        if self.laser_bckgrs:
            self.laser_bckgr = np.mean(np.array(self.laser_bckgrs), axis=0)
        else:
            self.laser_bckgr = None

    def _load_mask(self):
        """
        loads mask and converts to boolean values per pixel with the correct shape
        """
        if isinstance(self.mask_dir, PathLike):
            self.mask = np.load(self.mask_dir).T
            if self.mask.shape == self.dark_bckgr.shape:
                self.mask = self.mask.astype(bool)
            else:
                raise Exception(f"shape of mask and image do not match. \nshape mask: {self.mask.shape} \nshape dark_bckgr: {self.dark_bckgr.shape}")
        else:
            self.mask = np.ones(self.dark_bckgr.shape, dtype=bool)


    def _load_and_save_unpumped(self, 
                                correct_dark: bool = True, 
                                output_dir: PathLike = None, 
                                unpumped_fname: str = "unpumped.h5"):
        """
        Load all pumpoff files. Long and short if present
        Store metadata in self.file_registry and average one mean image: self.pumpoff_long and self.pumpoff_short each per short and long exposure time.
        Optional: Save all pumpoff images in a h5 file

        Parameters
        ----------
        correct_dark: bool, optional
            when True, every pumpoff image is subtracted by the mean self.dark_bckgr image
        output_dir: PathLike, optional
            when given, the single pumpoff images are saved in a separate h5 file (umpumped.h5) in the here given directory (same as the final h5 file)
        unpumped_fname: str, optional
            name of the saved h5 file with unpumped images when output_dir is given
                    
        Returns
        -------

        """
        if self.progress:
            print("loading pumpoffs")
            cycles = tqdm(self.cycles)
        else:
            cycles = self.cycles

        # go through all cycles and collect all pumpoff files 
        _pumpoff_long_pathlist = [] # list for all long pumpoff diectories for average image and reading all imgs to h5 file
        _pumpoff_short_pathlist = []
        for cyc in cycles:
            _cycle_path = self.basedir/ f"Cycle {int(cyc)}"

            # collect all pumpoff files and sort in short and long filepath lists
            for f in listdir(_cycle_path):
                fpath = join(_cycle_path, f)
                rel_fpath = join(f"Cycle {cyc}", f)
                if rel_fpath in self.ignored_files: #skip ignored files
                    continue 
                if "ProbeOnPumpOff" in f and f.endswith(".npy"): #select pumpoffs
                    if "short" in f: #short pumpoffs
                        _pumpoff_short_pathlist.append(fpath)
                    else: #long pumpoffs
                        _pumpoff_long_pathlist.append(fpath)

        #sort the pathlists via timestamp, therefore add files to registry:
        for fpath in _pumpoff_long_pathlist:
            self._add_to_registry(fpath)
        for fpath in _pumpoff_short_pathlist:
            self._add_to_registry(fpath)

        #now seperate registry entries and sort:
        long_entries = [e for e in self.file_registry if e["img_type"]=="unpumped_long"]
        short_entries = [e for e in self.file_registry if e["img_type"]=="unpumped_short"]

        long_entries = sorted(long_entries, key=lambda e: e["timestamp"])
        short_entries = sorted(short_entries, key=lambda e: e["timestamp"])

        if output_dir: #load imgs, finish filling file_registry and save as h5
            h5_path = join(output_dir, unpumped_fname)
            with h5py.File(h5_path, "w") as f:
                self.pumpoff_long = self._process_and_save_unpumped(
                    f, "long", long_entries, correct_dark
                )
                if short_entries:
                    self.pumpoff_short = self._process_and_save_unpumped(
                    f, "short", short_entries, correct_dark
                )
                
            if long_entries:
                df_long = pd.DataFrame(long_entries)
                #remove filepath (too long for HDF5 Table)
                df_long = df_long.drop(columns=['filepath'], errors='ignore')
                df_long.to_hdf(h5_path, key="long/metadata", mode="a")

            if short_entries:
                df_short = pd.DataFrame(short_entries)
                df_short = df_short.drop(columns=['filepath'], errors='ignore')
                df_short.to_hdf(h5_path, key="short/metadata", mode="a")
            
            if self.progress:
                print(f"saved {len(long_entries)} long and {len(short_entries)} short pumpoff images in h5 file")


        else: #load imgs in order to finish filling file_registry and save mean pumpoff
            if long_entries:
                self.pumpoff_long = self._only_process_unpumped("long", long_entries, correct_dark)

            if short_entries:
                self.pumpoff_short = self._only_process_unpumped("short", short_entries, correct_dark)

    def _only_process_unpumped(self, group_name: str, entries: list, correct_dark: bool):
        """
        loads each unpumped image in entries, updated registry and returns mean image

        Parameters
        ----------
        group_name: str
            "long" or "short"
        entries: list
            list of libraries from self.file_registry
        correct_dark: bool
            Whether unpumped images are corrected for dark_bckgr

        Return
        ------
        mean pumpoff: np.ndarray
            mean image of pumpoffs (either short or long)
        """
        n = len(entries)
        paths = [self.basedir/ entry["filepath"] for entry in entries]
        summed_img = None

        if self.progress:
            iterator = tqdm(paths, total=n, desc=f"Processing {group_name}")
        else:
            iterator = paths

        #load each image
        for fpath in iterator:
            _img = np.load(fpath).astype(np.float64).T
            if correct_dark:
                _img -= self.dark_bckgr
            _img = _img * self.mask

            # fill in missing data to registry
            for entry in entries:
                if entry["filepath"] == str(Path(fpath).relative_to(self.basedir)):
                    entry["total_intensity"] = _img.sum()
                    break

            #accumulate for mean img
            if summed_img is None:
                summed_img = _img
            else:
                summed_img += _img
        #calculate mean img
        if n > 0:
            mean_img = summed_img/n
        else:
            mean_img = np.zeros(self.dark_bckgr.shape)
        
        return( mean_img )

    def _process_and_save_unpumped(
            self,
            h5file: h5py.File,
            group_name: str,
            entries: list,
            correct_dark: bool
            ):
        """
        process unpumped images (either short or long) meaning filling in missing data in registry (img.sum, h5_index, ect),
        save images in h5 file and return mean
        
        Parameters
        ----------
        h5file: h5py.File
            file in which to save the images (either in key long or short)
            structure:
            unpumped.h5
            ├── long/
            │   ├── images           # (N, H, W) - HDF5 Dataset
            │   └── metadata         # Pandas DataFrame als HDF5 Table
            └── short/
                ├── images           # (N, H, W)
                └── metadata         # Pandas DataFrame
        group_name: str
            "long" or "short"
        entries: list
            list of libraries from self.file_registry
        correct_dark: bool
            Whether unpumped images are corrected for dark_bckgr

        Return
        ------
        mean pumpoff: np.ndarray
            mean image of pumpoffs (either short or long)
        """

        group = h5file.create_group(group_name)
        n = len(entries)
        paths = [self.basedir/entry["filepath"] for entry in entries]

        #create image dataset
        img_dset = group.create_dataset(
            "images",
            shape=(n,) + self.dark_bckgr.shape,
            dtype=np.float64,
            chunks=(1,) + self.dark_bckgr.shape,
            compression="gzip"
        )

        summed_img = None

        if self.progress:
            iterator = tqdm(enumerate(paths), total=n, desc=f"Processing {group_name} unpumped imgs")
        else:
            iterator = enumerate(paths)

        #load each image
        for i, fpath in iterator:
            _img = np.load(fpath).astype(np.float64).T
            if correct_dark:
                _img -= self.dark_bckgr
            _img = _img * self.mask
            
            # Save to h5
            img_dset[i] = _img #in order of timestamps since entries is in that order
            
            # fill in missing data to registry
            for entry in entries:
                if entry["filepath"]==str(Path(fpath).relative_to(self.basedir)):
                    entry["total_intensity"] = _img.sum()
                    entry["h5_index"] = i

            #accumulate for mean img
            if summed_img is None:
                summed_img = _img
            else:
                summed_img += _img
        #calculate mean img
        if n > 0:
            mean_img = summed_img/n
        else:
            mean_img = np.zeros(self.dark_bckgr.shape)
        
        return( mean_img )


    def _load_and_save_pumped(self, correct_laser: bool = True, save_diff: bool = True, correct_dark: bool = True):
        """
        Go through the cycles and collect and average the pumped images per delay step over the cycles
        
        Parameters
        ----------
        correct_laser: bool, optional
            if True the pumped images are subtracted by the averages self.laser_bckgr (SOON optional correct for different laser_bckgrs for each stage_position)
        save_diff : bool, optional
            if True, additional to the pumped images there will be averaged difference images with each pumped image being subtracted by the corresponding unpumped_long in labtime.                
        correct_dark: bool, optinal
            if True the unpumped images for the difference images are corrected by self.dark_bckgr
        """
        #prepare lists for counts of skipped/ignored delay times and averaged data
        self.missing_cycles_per_delay = np.zeros(len(self.delay_times)) # list with entry per delaystep, what happens when entry=#cycle (meaning no img for this delaytime)? order of entries is small to large stagepositions...not necessarily delaytimes.
        #list with cycle averaded pumped image per delaystep
        self.pumped_data = np.zeros(
            (len(self.delay_times), self.dark_bckgr.shape[0], self.dark_bckgr.shape[1])
        )
        #now for difference images
        self.missing_cycles_per_delay_diff = np.zeros(len(self.delay_times)) # list with entry per delaystep, later +1 if pumpoff OR pumped not there/ignored
        self.difference_data = np.zeros(
            (len(self.delay_times), self.dark_bckgr.shape[0], self.dark_bckgr.shape[1])
        )

        if self.progress:
            cycles = tqdm(self.cycles, desc="loading pumped images")
        else:
            cycles = self.cycles

        # load the actual data cycle for cycle
        for cycle in cycles:
            self._load_cycle_pumped(cycle, correct_laser=correct_laser, save_diff=save_diff, correct_dark=correct_dark)

        valid_counts = len(self.cycles) - self.missing_cycles_per_delay
        self.valid_delays_mask = valid_counts > 0 #bool array

        valid_counts_diff = len(self.cycles) - self.missing_cycles_per_delay_diff
        self.valid_delays_mask_diff = valid_counts_diff > 0 #bool array

        #normalize to considered amount of images
        for i in range(len(self.delay_times)):
            if self.valid_delays_mask[i]:
                self.pumped_data[i] /= valid_counts[i]
            else:
                self.pumped_data[i] = np.nan
                if self.progress:
                    print(f"No valid data for pumped data at delay {self.delay_times[i]:.2f} ps (position {self.stage_positions[i]})")

        if save_diff:
            #normalize to considered amount of images
            for i in range(len(self.delay_times)):
                if self.valid_delays_mask_diff[i]:
                    self.difference_data[i] /= valid_counts_diff[i]
                else:
                    self.difference_data[i] = np.nan
                    if self.progress:
                        print(f"No valid data for difference image at delay {self.delay_times[i]:.2f} ps (position {self.stage_positions[i]})")

        # here we sort the data so that small delay times are at low index values just for convenience,
        if self.delay_times[0] > self.delay_times[-1]: #depending on if small position on the stage corresponds to small or large delaytime
            self.delay_times = self.delay_times[::-1]
            self.stage_positions = self.stage_positions[::-1]
            self.pumped_data = self.pumped_data[::-1]
            self.difference_data = self.difference_data[::-1]

    def _load_cycle_pumped(self, cycle: int, correct_laser: bool = True, save_diff: bool = True, correct_dark: bool = True):
        """
        load pumped images (sorted by self.stage_positions, in _load_and_save_pumped it is sorted by self.delay_times)
        add cycle data per stage_position to self.pumped_data
        
        Parameters
        ----------
        cycle: int
            number of cycle from which the pumped data is collected
        correct_laser: bool, optional
            if True the pumped images are subtracted by the averages self.laser_bckgr (SOON optional correct for different laser_bckgrs for each stage_position)
        save_diff : bool, optional
            if True, additional to the pumped images there will be averaged difference images with each pumped image being subtracted by the corresponding unpumped_long in labtime.
        correct_dark: bool, optinal
            if True the unpumped images for the difference images are corrected by self.dark_bckgr
        """

        _cycle_path = self.basedir/ f"Cycle {int(cycle)}"
        _flist = listdir(_cycle_path)

        for _idx, pos in enumerate(self.stage_positions):
            _pos_flist = [] # collect filenames at this delaystage position in this cycle (=collect all frames)
            _pos_unpumped_flist = []
            _name = f"z_ProbeOnPumpOn_{str(pos).replace('.',',')} mm_Frm"
            for f in _flist:
                if _name in f:
                    if isinstance(self.ignored_files, list):
                        if f.endswith(".npy") and join(f"Cycle {cycle}", f) not in self.ignored_files:
                            _pos_flist.append(f) #add file if not in ignored_files
                    else: #if no files to be ignored were given
                        if f.endswith(".npy"):
                            _pos_flist.append(f)
                #collect unpumped frames for pos to be able to save difference images            
                if save_diff:
                    if _name.replace("ProbeOnPumpOn","ProbeOnPumpOff") in f:
                        if isinstance(self.ignored_files, list):
                            if f.endswith(".npy") and join(f"Cycle {cycle}", f) not in self.ignored_files:
                                _pos_unpumped_flist.append(f) #add file if not in ignored_files
                        else: #if no files to be ignored were given
                            if f.endswith(".npy"):
                                _pos_unpumped_flist.append(f)


            # if no file is collected, fill with zeros
            diff_flag = False # if a diff image can/should be saved for this position in this cycle
            if save_diff:
                diff_flag = True
                if not _pos_flist or not _pos_unpumped_flist:
                    diff_flag = False
                    self.missing_cycles_per_delay_diff[_idx] += 1

            if not _pos_flist:
                self.missing_cycles_per_delay[_idx] +=1

            else:
                #load images
                _pos_data = [] # list of 2d arrays (frames of pumped imgs)
                for f in _pos_flist:
                    fpath = join(_cycle_path, f)
                    _img = np.load(fpath).astype(np.float64).T
                    if correct_laser:
                        _img -= self.laser_bckgr
                    _img *= self.mask

                    _pos_data.append(_img)
                    self._add_to_registry(fpath, intensity=_img.sum())

                self.pumped_data[_idx] += np.mean(_pos_data, axis=0)

                if diff_flag:
                    _pos_unpumped_data = [] # list of 2d arrays (frames of unpumped imgs)
                    for f in _pos_unpumped_flist:
                        fpath = join(_cycle_path, f)
                        _img = np.load(fpath).astype(np.float64).T
                        if correct_dark:
                            _img -= self.dark_bckgr
                        _img *= self.mask

                        _pos_unpumped_data.append(_img)
                    
                    #save difference image which is normalized to unpumped image
                    self.difference_data[_idx] += (np.mean(_pos_data, axis=0) - np.mean(_pos_unpumped_data, axis=0)) / np.mean(_pos_unpumped_data, axis=0)

                        


    def _add_to_registry(self, fpath, intensity: float = None): #position somewhere else? later?
        """Helper to add entry to registry."""
        rel_fpath = Path(fpath).relative_to(self.basedir)
        f = Path(fpath).stem
        identifyer = findall(r"z_(\D+)\d+,\d+.*?_Frm\d+", str(f))[0]
        match identifyer:
            case "ProbeOffPumpOff_": #should not happen because not added to registry
                img_type = "dark_bckgr"
            case "ProbeOffPumpOn_": #should not happen because not added to registry
                img_type = "laser_bckgr"
            case "ProbeOnPumpOff_":
                img_type = "unpumped_long"
            case "ProbeOnPumpOff_short":
                img_type = "unpumped_short"
            case "ProbeOnPumpOn_":
                img_type = "pumped"
        cyc = int(Path(fpath).parts[-2].replace("Cycle ", ""))
        stage_pos = float(findall(r"z_\D+(\d+,\d+).*?_Frm\d+", str(f))[0].replace(",", "."))

        #stage_pos = float(f.split("_")[2].replace(",", ".").replace(" mm", ""))

        self.file_registry.append({
            "img_type": img_type,
            "cycle": cyc,
            "stage_position": stage_pos,
            "delay_time": self.delaytime_from_stageposition(stage_pos),
            "frame": int(f.split("Frm")[1].split(".")[0]),
            "filepath": str(rel_fpath),
            "total_intensity": intensity,
            "timestamp": self._read_timestamp_from_npyfpath(fpath),
            "h5_index": -1 #so that pandas can use int64
        })


    def _read_timestamp_from_npyfpath(self, fpath_npy: str) -> datetime:
        """
        Reads Unix-Timestamp (number of seconds since 01.01.1970) from accompanying .txt file to the .npy file

        Parameters
        ----------
        fpath_npy: str
            directory to the .npy file

        Returns
        -------
        float
            labtime of the taken image in Unix-Timestamp format
        """

        fpath_txt = fpath_npy.replace(".npy",".txt")
        with open(fpath_txt, "r") as f:
            return( int(findall(r"(?<=\\A2\\)\d+", f.readlines()[1])[0]) )


        
    def delaytime_from_stageposition(self, position):
        """
        converts delay stage position (mm) to delay time (ps)
        """
        speed_of_light_mm_per_ps = speed_of_light * 1e3 / 1e12
        return( 2 * (self.rel_pos_zero - position) / speed_of_light_mm_per_ps) #smalles delay time is arbitrarily set to 0 ps







    # 5. properties

    # 6. Dunder methods (besides __init__)
    def __repr__(self):
        return f"UEDDataset(basedir={str(self.basedir)}, cycles={self.cycles})"
    
    #def __len__(self):
        #return(len(self.cycles))
