"""
Nonsense filler for now
"""

from typing import Union, Literal, Tuple
import numpy as np
from PIL import Image
from os import PathLike, listdir
from os.path import join, isfile
from re import findall
from scipy.constants import speed_of_light
from scipy.ndimage import median_filter
from tqdm import tqdm
from datetime import datetime
from lmfit.model import Model, ModelResult
import h5py


class UnpumpedDataset:
    def __init__(self, path: PathLike, background: Union[PathLike, None] = None):
        """
        generate static dataset from folder. perform background correction if background file is given

        Parameters
        ----------
        path : PathLike
            _description_
        background : Union[PathLike, None], optional
            _description_, by default None
        """

        self.path = path
        self.background_file = background

        filelist = [
            entry for entry in listdir(self.path) if isfile(join(self.path, entry))
        ]

        _raw_imgs = []
        self.imgs = []

        for file in filelist:
            if background:
                if file.endswith(".tif") and file != background:
                    _raw_imgs.append(self._load_image(join(self.path, file)))
                elif file == background:
                    self.background = self._load_image(join(self.path, file))

            else:
                if file.endswith(".tif"):
                    self.imgs.append(self._load_image(join(self.path, file)))

        if background:
            for img in _raw_imgs:
                self.imgs.append(img - self.background)

        self.mean = np.mean(np.array(self.imgs), axis=0)