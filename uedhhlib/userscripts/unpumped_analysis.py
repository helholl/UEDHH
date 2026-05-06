"""
Analysis of the unpumped images of a UED measurement in order to trace ebeam count and crystal degredation over labtime

unpumped data must be given as hdf5 files and a poni file containing the center of the diffraction rings (pyfai-calib2)
"""

import numpy as np
import h5py
import pyFAI
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

analysis_folder = r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2_hpxrm\hlh_analysis"
poni_name = "UEDPerylExp14_cyc1.poni"
mask_name = "peryl_exp14_mask.npy"
mainbeam_mask = "peryl_exp14_mainbeam_mask.npy"
unpumpedH5_name = "exp14_unpumped_woarcs_hpxrm.h5"
pdf_name = "Peryl_exp14_hpxrm_azigral_unpumped.pdf"

mb_mask = ~np.load(join(analysis_folder, mainbeam_mask)).astype("bool")
poni = pyFAI.load(join(analysis_folder, poni_name))
mask = np.load(join(analysis_folder, mask_name))
FIGSIZE = (16,9)
CHUNK_SIZE = 50
#mb_intensities = []
#long_lineouts = []

with h5py.File(join(analysis_folder, unpumpedH5_name)) as h:
    longs = h["long/images"]
    shorts = h["short/images"]
    first_short = shorts[0]
    first_long = longs[0]
    long_registry = pd.read_hdf(join(analysis_folder, unpumpedH5_name), key="long/metadata")
    short_registry = pd.read_hdf(join(analysis_folder, unpumpedH5_name), key="short/metadata")

    # n_longs = long_registry.size
    # for start in tqdm(range(0, n_longs, CHUNK_SIZE)):
    #     end = min(start+CHUNK_SIZE, n_longs)
    #     chunk = shorts[start:end]
    #     for img in chunk:
    #         _, lo = poni.integrate1d(img,  npt=1024, mask=mask, unit="q_A^-1", radial_range=(1, 9))
    #         long_lineouts.append(lo)

#long_lineouts = np.array(long_lineouts)
#np.save(join(analysis_folder, "long_lineouts.npy"), long_lineouts)
long_lineouts = np.load(join(analysis_folder, "long_lineouts.npy"))
qs, first_long_lo = poni.integrate1d(first_long,  npt=1024, mask=mask, unit="q_A^-1", radial_range=(1, 9))

    # n_images = short_registry.size
    # for start in tqdm(range(0, n_images, CHUNK_SIZE)):
    #     end = min(start+CHUNK_SIZE, n_images)
    #     chunk = shorts[start:end]
    #     chunk_mb = chunk * mb_mask
    #     chunk_ints = chunk_mb.sum(axis=(1,2))
    #     chunk_ints = [int(s) for s in chunk_ints]
    #     mb_intensities.extend(chunk_ints)

#mb_intensities = np.array(mb_intensities)
#np.save(join(analysis_folder, "short_mainbeam_intensities.npy"), mb_intensities)
mb_intensities = np.load(join(analysis_folder, "short_mainbeam_intensities.npy"))
timestamps_short = short_registry["timestamp"].values
labtime_short = pd.to_datetime(timestamps_short, unit="s")
timestamps_long = long_registry["timestamp"].values
labtime_long = pd.to_datetime(timestamps_long, unit="s")

#calculate the transient intensity of a chosen peak in the long lineouts
peak = 7.11 #in 1/Angstr
peak_idx = np.abs(np.array(qs)-peak).argmin()
width = 0.2 #in 1/Angstr
roi = (qs>qs[peak_idx]-width) & (qs<qs[peak_idx]+width)

peak_intensities = []
for lo in long_lineouts:
    p_int = sum(lo[roi])
    peak_intensities.append(p_int)
peak_intensities = np.array(peak_intensities)

with PdfPages(join(analysis_folder, pdf_name)) as pdf:
########################################################################
    #plot 2D imgs

    fig, axs = plt.subplots(1,2, figsize=FIGSIZE)
    fig.suptitle("Unpumped Analysis: First Images")
    axs = axs.flatten()

    axs[0].set_title("First unpumped short * mask")
    im0 = axs[0].imshow(first_short*mb_mask, 
                        cmap="inferno", 
                        vmin=0, 
                        vmax=np.percentile(first_short*mb_mask, 99.95))
    im1 = axs[0].imshow(mb_mask, cmap="PiYG", alpha=0.9)
    fig.colorbar(im0, ax=axs[0])

    axs[1].set_title("First unpumped long")
    im1 = axs[1].imshow(first_long, 
                        cmap="inferno", 
                        vmin=0, 
                        vmax=np.percentile(first_long, 99))
    fig.colorbar(im1, ax=axs[1])

    fig.tight_layout()
    pdf.savefig(fig)
#########################################################################
    #plot long analysis
    print(labtime_long.shape)
    print(peak_intensities.shape)

    fig, axs = plt.subplots(2,1, figsize=FIGSIZE)
    fig.suptitle("Unpumped Analysis: Transient in Labtime")
    axs = axs.flatten()

    axs[0].set_title("Azimuthal integral of first unpumped long")
    axs[0].plot(qs, first_long_lo, color="gray")
    axs[0].axvspan(qs[roi].min(), qs[roi].max(), alpha=0.3, color="orange", lw=0)
    axs[0].axvline(qs[peak_idx], color="orange")
    axs[0].set_yscale('log')
    axs[0].set_ylabel("Intensity /a.u.")
    axs[0].set_xlabel(r"Scattering vector q /$\mathrm{\AA}^{-1}$")

    axs[1].set_title("Summed intensities over labtime")
    axs[1].scatter(labtime_short, mb_intensities, color="blue", label="sum(unpumped short * mask)")
    axs[1].tick_params(axis="y", labelcolor="blue")
    axs[1].set_xlabel("Labtime")
    axs[1].set_ylabel("Summed Mainbeam Intensity /a.u.", color="blue")
    axs[1].legend()

    axs2 = axs[1].twinx()
    axs2.scatter(labtime_long, peak_intensities[:-1], color="orange", label="sum over roi")
    axs2.tick_params(axis="y", labelcolor="orange")
    axs2.set_ylabel("Peak Intensity /a.u.", color="orange")
    axs2.legend(loc='upper right', bbox_to_anchor=(1, 0.92))

    fig.tight_layout()
    pdf.savefig(fig)
