from typing import Union, Tuple, List
import numpy as np
from os import PathLike, listdir
from os.path import join
from datetime import datetime


class BaseDataset:
    def __init__(
            self,
            basedir: PathLike,
            progress: bool = True,
            mask: np.ndarray = None,
            correct_background: bool = True,
            cycles: Union[int, tuple, List[int]] = None,
            ignore: list = None          
            ):
        """
        Common functionality for all datasets(pumped, unpumped, static)

        Parameters
        ---------
        basedir : PathLike
            base directory cotaining the "Cycle X" directories
        progress : bool, optional
            turn on progress notification during data loading, by default True
        mask : np.ndarray, optional
            constant mask applied to all loaded images. Mask should be array of bool values where False values mark used points. If 'None' all points of the images are used, by default None
        correct_background : bool, optional
            wether laser background is corrected for or not, by default True
        cycles : Union[int, tuple, List[int]], optional
            For now, give this parameter an iterable containing the cycle number you want to load, e.g: (1,2,4,7,10,11,12) to load cycles 1,2,4.... you get it
        ignore : list
            list of tuples of the form (cycle_number, stage_position as string, frame number). To ignore frames three of cycle 5 at stage position 105.4 mm use (5, 105.4, 3).
            Note: Make sure that stage position is given with correct decimal separator "."
        """
        
        self.basedir = basedir
        self.progress = progress        
        self.mask = mask
        self.correct_background = correct_background

        #always get list of cycles
        if isinstance(cycles, list):
            self.cycles = cycles
        elif isinstance(cycles, int):
            self.cycles = [cycles]
        elif cycles is None:
            raise Exception("No cycles entered")
        else:
            self.cycles = list(cycles)

        #handle ignore list
        if ignore:
            self.ignore = ignore
            self._make_ignored_files_list()
        else:
            self.ignore = []
            self.ignored_files = []

        #metadata storage
        self.loaded_files = []
        self.timestamps = []
        self.realtime_intensities = []

        #background storage
        self.dark_bckgrs = [] #list of all ProbeOffPumpOff images
        self.dark_bckgr = None #average ProbeOffPumpOff image over all cycles

        #will be set by child classes
        self.data = None

    def _build_filename(self, cycle: int, stage_pos: float, frame: int, 
                        pattern: str) -> str:
        """
        Build filename from components
        
        Parameters
        ----------
        cycle : int
            Cycle number
        stage_pos : float
            Stage position in mm with "." as separator
        frame : int
            Frame number
        pattern : str
            Pattern like "ProbeOnPumpOn_", "ProbeOnPumpOff_", "ProbeOnPumpOff_short", 
            "ProbeOffPumpOn_", "ProbeOffPumpOff_"
        
        Returns
        -------
        str
            Full file path
        """
        filename = f"z_{pattern}{str(stage_pos).replace('.', ',')} mm_Frm{int(frame)}.npy"
        filepath = join(self.basedir, f"Cycle {int(cycle)}", filename)
        return filepath


    def _make_ignored_files_list(self, pattern: str):
        """
        Makes a list of files to be ignored during loading from list of touples given by datapicker.py
        
        Parameters
        ----------
        pattern : str
            Filename pattern (e.g., "ProbeOnPumpOn_", "ProbeOnPumpOff_")
        """
        self.ignored_files = []
        for cycle, stage_pos, frame in self.ignore:
            filepath = self._build_filename(cycle, stage_pos, frame, pattern)
            self.ignored_files.append(filepath)

        
    def _load_dark_bckgr(self):
        """
        Loads background data (ProbeOffPumpOff) from all cycles and create mean image.
        Used by all dataset types for shape reference.
        Unpumped data is is corrected by the background.
        """
        for cycle in self.cycles:
            _cycle_path = join(self.basedir, f"Cycle {int(cycle)}")
            _dark_bckgr_list = []
            for file in listdir(_cycle_path):
                if "ProbeOffPumpOff" in file and file.endswith(".npy"):
                    _dark_bckgr_list.append(join(_cycle_path, file))

            for dark_bckgr in sorted(_dark_bckgr_list):
                dark_bckgr_img = np.load(dark_bckgr)
                self.dark_bckgrs.append(dark_bckgr_img)

        if self.dark_bckgrs:
            self.dark_bckgr = np.mean(np.array(self.dark_bckgrs), axis=0)
        else:
            self.dark_bckgr = None

