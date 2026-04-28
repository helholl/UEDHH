### This tool is adapted from Laurenz Kremeyer, Siwick group, McGill University, Montreal
import sys
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QKeySequence
from pyqtgraph.Qt import QtWidgets, QtGui
import numpy as np
from re import findall
from datetime import datetime
import json
from pathlib import Path

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "filepath", help="filepath to h5 file saved from Dataset from uedhhlib"
    )
    args = parser.parse_args()
    return args


class DataPicker(QtWidgets.QMainWindow):
    def __init__(self, df, default_dir: str ="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        df = df.sort_values("timestamp").reset_index(drop=True)

        self.timestamps = df["timestamp"].values
        #self.intensities = df["total_intensity"].values
        self.loaded_files = df["filepath"].values
        self.df = df
        self.default_dir = default_dir

        self.lris = []
        self.shortcut_add_lri = QtGui.QShortcut(QtGui.QKeySequence("+"), self)
        self.shortcut_add_lri.activated.connect(self.add_lri)
        self.shortcut_remove_lri = QtGui.QShortcut(QtGui.QKeySequence("-"), self)
        self.shortcut_remove_lri.activated.connect(self.remove_lri)
        self.shortcut_print_rule = QtGui.QShortcut(QtGui.QKeySequence("space"), self)
        self.shortcut_print_rule.activated.connect(self.print_rule)
        self.shortcut_save_ignored = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.shortcut_save_ignored.activated.connect(self.save_ignored)


        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graphics_layout.addPlot(
            axisItems={"bottom": pg.DateAxisItem()}
            )
        
        #plot pumped, unpumped_long and unpumped_short separately
        pumped = df[df["img_type"]=="pumped"]
        unpumped_long = df[df["img_type"]=="unpumped_long"]
        unpumped_short = df[df["img_type"]=="unpumped_short"]
        dark_bckgr = df[df["img_type"]=="dark_bckgr"]
        laser_bckgr = df[df["img_type"]=="laser_bckgr"]

        legend = self.plot.addLegend(offset=(-10, 10))
        legend.setBrush(pg.mkBrush(50, 50, 50, 200)) #dark background, slightly transparent
        legend.setPen(pg.mkPen(color="w", width=1)) #white bow
        self.plot.plot(
            x=pumped["timestamp"].values,
            y=pumped["total_intensity"].values,
            pen=pg.mkPen(color="#ffa500", width=1), #None: no line, style=Qt.DashLine
            symbol="d",
            symbolsize=3,
            symbolBrush="#ffa500", #color (orange)
            name="pumped"
        )
        self.plot.plot(
            x=unpumped_long["timestamp"].values,
            y=unpumped_long["total_intensity"].values,
            pen=pg.mkPen(color="b", width=1), #no line
            symbol="d",
            symbolsize=3,
            symbolBrush="b", #color (blue)
            name="unpumped (long)"
        )
        self.plot.plot(
            x=unpumped_short["timestamp"].values,
            y=unpumped_short["total_intensity"].values,
            pen=pg.mkPen(color="g", width=1), #no line
            symbol="d",
            symbolsize=3,
            symbolBrush="g", #color (green)
            name="unpumped (short)"
        )
        self.plot.plot(
            x=dark_bckgr["timestamp"].values,
            y=dark_bckgr["total_intensity"].values,
            pen=pg.mkPen(color="#38038f", width=1), #no line
            symbol="d",
            symbolsize=3,
            symbolBrush="#38038f", #color (green)
            name="dark background"
        )
        self.plot.plot(
            x=laser_bckgr["timestamp"].values,
            y=laser_bckgr["total_intensity"].values,
            pen=pg.mkPen(color="#f5ff9c", width=1), #no line
            symbol="d",
            symbolsize=3,
            symbolBrush="#f5ff9c", #color (green)
            name="laser background"
        )

        #self.curve = self.plot.plot(x=timestamps, y=intensities)
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.setCentralWidget(self.graphics_layout)
        self.show()

    def __len__(self):
        return len(self.timestamps)

    @pyqtSlot()
    def add_lri(self):
        total_duration = self.timestamps[-1] - self.timestamps[0]
        width = int(total_duration / 10) #10% of the measurement duration
        if not self.lris:
            mi = self.timestamps[0]
            ma = mi + width
        else:
            mi = self.lri_regions[-1][-1] + int(width / 2)
            ma = mi + width
        lri = pg.LinearRegionItem([mi, ma])
        self.lris.append(lri)
        self.plot.addItem(lri)

    @pyqtSlot()
    def remove_lri(self):
        pass

    @pyqtSlot()
    def print_rule(self):
        output = "ignored_files = ["
        for region in self.lri_regions:
            selected = self.df[
                (self.df["timestamp"] >= region[0]) &
                (self.df["timestamp"] <= region[1])
            ]

            for _, row in selected.iterrows():
                output += f"({int(row["cycle"])}, {row["img_type"]}, {float(row["stage_position"])}, {int(row["frame"])}), "

        # remove last space and comma separator and close list
        output = output.rstrip(", ") + "]"
        print(output)

    @pyqtSlot()
    def save_ignored(self):
        selected_rows = []
        for region in self.lri_regions:
            selected = self.df[
                (self.df["timestamp"] >= region[0]) &
                (self.df["timestamp"] <= region[1])
            ]
            selected_rows.append(selected)

        if not selected_rows:
            return #nothing selected, nothing gets saved
        
        all_selected = pd.concat(selected_rows)

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save ignored labels", #tab name
            str(Path(self.default_dir)/"ignored_labels.json"), #proposed filename
            "JSON Files (*.json)" #datafilter
        )

        if not filename:
            return #user canceled
        
        #ignored labels as list of tuples
        ignored = []
        for _, row in all_selected.iterrows():
            ignored.append({
                "cycle": int(row["cycle"]),
                "img_type": row["img_type"],
                "stage_position": float(row["stage_position"]),
                "frame": int(row["frame"]),
                "filepath": row["filepath"]
            })
        
        with open(filename, "w") as f:
            json.dump(ignored, f, indent=4)

        print(f"saved {len(ignored)} ignored files to {filename}")


    @property
    def lri_regions(self):
        return [
            (int(np.rint(lri.getRegion()[0])), int(np.rint(lri.getRegion()[1])))
            for lri in self.lris
        ]


if __name__ == "__main__":
    import sys
    import pandas as pd
    
    args = parse_args()
    df = pd.read_hdf(args.filepath, key="file_registry")

    app = QtWidgets.QApplication(sys.argv)
    win = DataPicker(df, default_dir=str(Path(args.filepath).parent))
    sys.exit(app.exec())
