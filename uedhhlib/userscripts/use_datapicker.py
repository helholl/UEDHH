from uedhhlib.tools import DataPicker
import sys
import pandas as pd
from pyqtgraph.Qt import QtWidgets
from pathlib import Path

raw_h5_fpath = r"Z:\Users\Emma\Zyla1\2026\2026_11\NT52\261117\Au\Meas3\hlh_analysis\exp16_raw.h5"

df = pd.read_hdf(raw_h5_fpath, key="file_registry")

app = QtWidgets.QApplication(sys.argv)
win = DataPicker(df, default_dir=str(Path(raw_h5_fpath).parent))
sys.exit(app.exec())