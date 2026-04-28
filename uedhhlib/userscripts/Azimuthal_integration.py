import numpy as np
import pyFAI
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import getcwd
from datetime import datetime
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from skued import biexponential, with_irf
from scipy.optimize import curve_fit
import dask.array as da
import h5py

from uedhhlib.datasets import PumpedDataset
from uedhhlib.analysis import find_center, colors_from_arr

CMAP = plt.get_cmap("inferno")
FIGSIZE = (16, 9)
T0 = 10
COLS = 3
#MASK = np.load(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis\peryl_exp14_mask.npy").astype(bool)

# Stelle sicher, dass deine Maske nicht alle Werte entfernt
# print("Maske Shape:", MASK.shape)
# print("MASKe Summe:", MASK.sum())
# print("MASKe Prozent:", MASK.sum() / MASK.size * 100)
# print("Maske Typ:", MASK.dtype)

def color_enumerate(iterable, start=0, cmap=CMAP):
    """
    same functionality as enumerate, but additionally yields sequential colors from
    a given cmap
    """

    n = start
    try:
        length = len(iterable)
    except TypeError:
        length = len(list(iterable))
    for item in iterable:
        yield n, cmap(n / (length - 1)), item
        n += 1

with h5py.File(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis\exp14_25cyc_woarcs.h5") as h:
    pumped_data = h["processed/intensity"][()]
    long_pumpoff = h["processed/equilibrium"][()]
    print(type(long_pumpoff))
    delays = h["time_points"][()]

    pumped_data = np.transpose(pumped_data, (2,0,1))
    AI = pyFAI.load(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis\UEDPerylExp14_cyc1.poni")

    BINS = 1024
    INTEGRATION_WIDTH = 0.05
    INTEGRATION_KWARGS = dict(mask=None, unit="q_A^-1", radial_range=(2, 8))

    T0_idx = np.abs(np.array(delays)-T0).argmin()
    qs, lineout_eq = AI.integrate1d(
        np.mean(pumped_data[:T0_idx+1], axis=0), BINS, **INTEGRATION_KWARGS
    )
    NORM_RANGE = (qs > 2.5) & (qs < 7)
    lineout_eq /= np.mean(lineout_eq[NORM_RANGE])
    qs, lineout_mean = AI.integrate1d(
        np.mean(pumped_data, axis=0), BINS, **INTEGRATION_KWARGS
    )
    lineout_mean /= np.mean(lineout_mean[NORM_RANGE])
    ls = []
    for img in pumped_data:
        qs, lo = AI.integrate1d(img, BINS, **INTEGRATION_KWARGS)
        lo /= np.mean(lo[NORM_RANGE])
        ls.append(lo)
    ls = np.array(ls)

    with PdfPages(r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2\hlh_analysis\Peryl_exp14_anaysis.pdf") as pdf:
        try:
            f, ax = plt.subplots(1, 2, figsize=FIGSIZE)
            f.suptitle(
            f"dirname: {getcwd()}   T0: {T0}ps?\n"
        )
            ax = ax.flatten()

            im0 = ax[0].imshow(
                long_pumpoff, cmap="inferno", vmin=0, vmax=np.percentile(long_pumpoff, 99), norm="log"
            )
            ax[0].set_title("mean pump off")
            # im0
            divider0 = make_axes_locatable(ax[0])
            cax0 = divider0.append_axes("right", size="5%", pad=0.1)  # Breite 5%, Abstand 0.1
            cbar0 = f.colorbar(im0, cax=cax0)
            cbar0.set_label("Counts", fontsize=20, labelpad=0)
            cbar0.ax.tick_params(which='major', length=10, width=1.5, direction='inout')  # tick Länge 10, Breite 2
            cbar0.ax.tick_params(which='minor', length=6, width=1.5, direction='inout')
            for lbl in cbar0.ax.get_yticklabels():
                lbl.set_fontsize(20)

            im1 = ax[1].imshow(
                (np.mean(pumped_data[-10:], axis=0) - np.mean(pumped_data[:T0_idx], axis=0))
                / np.mean(pumped_data[:T0_idx], axis=0),
                cmap="bwr",
                vmin=-0.105,
                vmax=0.105,
            )
            ax[1].set_title("(post T0 - pre T0)/pre T0")
            # im1
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            cbar1 = f.colorbar(im1, cax=cax1)
            cbar1.set_label(r"$\Delta\mathrm{I}$ /%$\,\mathrm{I}_0$", fontsize=20, labelpad=0)
            # Beispiel: Originalwerte von -0.10 bis 0.10 sollen als -10 bis 10 angezeigt werden
            tick_vals = np.linspace(-0.10, 0.10, 5)  # 5 Ticks
            cbar1.set_ticks(tick_vals)
            cbar1.set_ticklabels([str(int(v*100)) for v in tick_vals])  # multipliziere mit 100
            cbar1.ax.tick_params(which='major', length=10, width=1.5, direction='inout') 
            cbar1.ax.tick_params(which='minor', length=6, width=1.5, direction='inout')  # tick Länge 10, Breite 2
            for lbl in cbar1.ax.get_yticklabels():
                lbl.set_fontsize(20)
        

            for i in range(2):
                ax[i].axis("off")

            f.tight_layout()
            # plt.show()
            pdf.savefig(f)

        except ValueError:
            pass
