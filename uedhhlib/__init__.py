#core datasets
from .datasets.pumped import PumpedDataset
from .datasets.unpumped import UnpumpedDataset
from .datasets.static import StaticDataset

#important processing functions
from .processing.hotpixels import hotpixel_filter

__all__ = [
    #datasets
    'PumpedDataset', 
    'UnpumpedDataset', 
    'StaticDataset',
    #processing
    'hotpixel_filter'
    ]


__version__ = '0.1.0'