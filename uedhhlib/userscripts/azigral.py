"""
Analysis of a UED datset of a polycrystalline sample. 

Data must be given as hdf5 file 
and a .poni file must be given, containing the center of the diffraction rings (pyfai-calib2)
Different diagnostics are shown in resulting pdf
"""

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import pyFAI
import h5py
from pathlib import Path
from datetime import datetime
from skued import biexponential, with_irf
from scipy.optimize import curve_fit
from uedhhlib.analysis.q2_weights import compute_q2_weight

analysis_folder_path = Path(r"Z:\Users\Emma\Zyla1\2026\2026_06\NT49\260619\NT49_Pos5_Perylene\Meas1_wohtpx\hlh_analysis")
poni_name = "UEDPerylExp16_meanlongpumpoff.poni"
mask_name = "peryl_exp16_mask_transp.npy"
pumpedH5_name = "exp16_woarcs1stAnd2ndIteration_transposed.h5" #_transposed
pdf_name = "Peryl_exp16_azigral_transp_q2weighted.pdf"

# set some variables
t0_set = 35 #in ps, guess set by hand due to knowledge of measurement, for time axis setting
fit_flag = False #set whether the transients shall be fitted
#insert peaks (line 104) after first time azigral
figsize = (16,9)
pre_t0 = -10 # in ps compared to t0, for pre t0 mean normalization
n_post = 6 # number of images considered for post t0 mean
norm_range_min = 4 #range in which the data is comparably constant i guess?
norm_range_max = 7
integr_width = 0.05 # width around which the peaks are integrated to get transients
radial_range=(0.6, 6)
xlim_min, xlim_max = 0.6, 6.2

############################################
# fit function for transients
def irf_fit(time, *args, **kwargs):
    """
    fit function for biexponendtial transients. Use skued biexp. fit

    args: tuple 
        (time_resolution, t0_fit, amp1, amp2, tconst1, tconst2, offset)

    """
    args = list(args)
    #args[4] = 0.35  # set the decay time constant
    # args[0] = 0.01  # set the IRF
    args = tuple(args)
    return with_irf(np.abs(args[0]))(biexponential)(time, *args[1:], **kwargs)
############################################

# load data 
with h5py.File(analysis_folder_path/ pumpedH5_name) as h:
    # pumped img per delaystep
    pumped_data = h["processed/intensity"][()]
    pumped_data = np.transpose(pumped_data, (2, 0, 1)) # to get dimension(delay_times, x, y)
    # mean long pumpoff img
    long_pumpoff = h["processed/equilibrium"][()]
    # delaysteps
    delay_times = h["time_points"][()]
    # files registry with metadata for all taken images during the measurement
    file_registry = pd.read_hdf(analysis_folder_path/ pumpedH5_name, key="file_registry")
    # timestamps for metadata for what images?

#save long_pumpoff:
#np.save(analysis_folder_path/"mean_long_pumpoff.npy", long_pumpoff)

print("all data is loaded\n")
delay_times = delay_times - t0_set
# load poni file and mask
poni = pyFAI.load(analysis_folder_path/ poni_name)
fit2d_geom = poni.getFit2D() 
fit2d_geom["directDist"] = 187  ## in mm 
poni.setFit2D(**fit2d_geom)
q2_weight = compute_q2_weight(long_pumpoff, analysis_folder_path/ poni_name) #np.array with factor per pixel

centerX, centerY = fit2d_geom["centerX"], fit2d_geom["centerY"]
mask = np.load(analysis_folder_path/ mask_name)

# calculate and format the measurement and analysis time: 
timestamps = file_registry["timestamp"].values
meas_start = timestamps.min()
start_dt = pd.to_datetime(meas_start, unit="s")
start_formatted = start_dt.strftime("%d.%m.%y %H:%M")
meas_end = timestamps.max()
end_dt = pd.to_datetime(meas_end, unit="s")
end_formatted = end_dt.strftime("%d.%m.%y %H:%M")
now = datetime.now().replace(microsecond=0)
now_formatted = now.strftime("%d.%m.%y %H:%M")

# set color map stuff
cm_norm = Normalize(vmin=delay_times.min(), vmax=delay_times.max())
cmap = plt.get_cmap("inferno")
#define binary colormap
c_transparent = mcolors.to_rgba('black', alpha=0)
c_opaque = mcolors.to_rgba('yellow', alpha=0.4)
binary_cmap = mcolors.LinearSegmentedColormap.from_list('binary_transparent', [c_transparent, c_opaque], N=2)

# calculate pumped mean images before and after t0
pre_idx = np.abs(np.array(delay_times) - pre_t0).argmin()
mean_before = np.mean(pumped_data[:pre_idx], axis=0)
mean_after = np.mean(pumped_data[-n_post:], axis=0)
# for division no pixel should be zero in mean_before
zero_mask = np.abs(mean_before) > 1e-10 #bool array true for values not zero
diff = np.full_like(mean_before, np.nan) #create array with only NaN entries
diff[zero_mask] = (mean_after - mean_before)[zero_mask] / mean_before[zero_mask] #fill in valid division pixels
#find index of t0
#t0_idx = np.abs(np.array(delay_times) - t0_set).argmin()

# integrate and collect lineouts, normalize to pumpoff lineout?
# qs: list of all q-values (same for every integrated image), lo: lineout
qs, lo_pumpoff = poni.integrate1d(long_pumpoff, npt=1024, mask=mask, unit="q_A^-1", radial_range=radial_range)
_, lo_mean_pumped =  poni.integrate1d(np.mean(pumped_data, axis=0), npt=1024, mask=mask, unit="q_A^-1", radial_range=radial_range)
_, lo_mean_before = poni.integrate1d(mean_before, npt=1024, mask=mask, unit="q_A^-1", radial_range=radial_range)
#range in which the data is comparably constant, so it can be used to compare imgs which are normalized to the mean value in this range
norm_range = (qs > norm_range_min) & (qs < norm_range_max) 

#normalize to mean value of lo within norm_range to be able to compare relative changes within lo
lo_pumpoff_norm = lo_pumpoff / np.mean(lo_pumpoff[norm_range]) 
lo_mean_pumped_norm = lo_mean_pumped / np.mean(lo_mean_pumped[norm_range])
lo_mean_before_norm = lo_mean_before / np.mean(lo_mean_before[norm_range])

los_norm = [] #list of lineout 1d arrays 
for delay_img in pumped_data:
    _, lo = poni.integrate1d(delay_img, npt=1024, mask=mask, unit="q_A^-1", radial_range=radial_range)
    lo /= np.mean(lo[norm_range])
    los_norm.append(lo)
los_norm = np.array(los_norm)
los_rel = (los_norm - lo_mean_before_norm) / lo_mean_before_norm #list of the changes compared to before t0

print("all data is integrated")

# for now identify peak position per hand. maybe peakfinder later
# peaks = [2.80, 4.45, 5.65, 6.00, 6.30, 6.95, 7.18, 8.80] #in inverse angstrom
# weird scale, therefore:
peaks = [0.82, 1.3, 1.66, 1.825, 2.09, 2.575, 2.88, 3.32, 3.69, 4.55, 5.35] #in inverse angstrom
peaks_idx = [] #index in qs list
for p in peaks:
    p_idx = np.abs(np.array(qs) - p).argmin()
    peaks_idx.append(p_idx)

rois = [(qs>qs[p]-integr_width) & (qs<qs[p]+integr_width) for p in peaks_idx]

with PdfPages(analysis_folder_path/ pdf_name) as pdf:
#########################################################################################
    # plot mean pumpoff and difference img 
    fig, axs = plt.subplots(1,2, figsize=figsize)
    fig.suptitle(f"HDF5 file: {pumpedH5_name}, azigral analysis: {now_formatted}, measuring time: {start_formatted} to {end_formatted}"), 
    axs = axs.flatten() # make list of array

    axs[0].set_title("Mean long_pumpoff (weighted by q2)")
    im0 = axs[0].imshow(long_pumpoff*q2_weight, cmap=cmap, vmin=0, vmax=np.percentile(long_pumpoff*q2_weight, 99))
    axs[0].imshow(mask, cmap=binary_cmap)
    axs[0].scatter(centerX-0.5, centerY-0.5, marker="x", s=20, color="g", label="Beam Center")
    axs[0].legend()
    fig.colorbar(im0, ax=axs[0])

    axs[1].set_title(f"(last {n_post} delays - first {pre_t0} ps) / first {pre_t0} ps")
    im1 = axs[1].imshow(diff, cmap="bwr", vmin=-0.05, vmax=0.05)
    axs[1].imshow(mask, cmap=binary_cmap)
    fig.colorbar(im1, ax=axs[1])

    for ax in axs:
        ax.axis("off")

    fig.tight_layout()
    pdf.savefig(fig)

#########################################################################################
    # plot one lineout with good readable peaks (add simulation??)
    simulation = np.load(analysis_folder_path/"perylene_sim_powderpattern.npy")
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Static (unpumped) lineout")

    for p, roi in zip(peaks_idx, rois):
        ax.axvspan(qs[roi].min(), qs[roi].max(), alpha=0.3, color="k", lw=0)
        ax.axvline(qs[p], color="k")

    ax.plot(qs, lo_pumpoff, label="Pumpoff lineout")
    ax.plot(qs, 0.2*lo_pumpoff*qs**2, label="Pumpoff lineout (q2-weighted)*0.2")
    q = np.linspace(0,6,5000)
    ax.plot(q, simulation*9e1, label="Simulated powderpattern \n(h, k and l from -8 to +8)")
    ax.set_xlabel(r"Scattering vector q /$\mathrm{\AA}^{-1}$")
    ax.set_ylabel("Intensity /a.u.")
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(0, 1000)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)


#########################################################################################
    # plot the integrated lineouts
    gs = gridspec.GridSpec(3, 2, width_ratios=[0.97, 0.03], height_ratios=[1, 1, 1])

    fig, axs = plt.subplots(3,1, sharex=True, figsize=figsize)
    fig.suptitle("Overview lineouts")
    axs = axs.flatten()

    axs[0].set_title(fr"Normalized lineouts (norm_range: {norm_range_min}-{norm_range_max} $\mathrm{{\AA}}^{{-1}}$) ")
    axs[1].set_title("Change lineout compared to mean long_pumpoff")
    axs[2].set_title(f"Change lineout compared to mean first {pre_t0} ps")

    for p, roi in zip(peaks_idx, rois):
            for ax in axs:
                ax.axvspan(qs[roi].min(), qs[roi].max(), alpha=0.3, color="k", lw=0)
                ax.axvline(qs[p], color="k")

    for idx, lo in enumerate(los_norm):
        c = cmap(cm_norm(delay_times[idx])) #color proportional to delaytime
        axs[0].plot(qs, lo*0.2*qs**2, color=c)
        axs[1].plot(qs, (lo - lo_pumpoff_norm) / lo_pumpoff_norm, color=c)
        axs[2].plot(qs, los_rel[idx], color=c)

    #axs[0].set_yscale('log')
    axs[2].set_xlabel(r"Scattering vector q /$\mathrm{\AA}^{-1}$")
    axs[0].set_ylabel("Intensity /a.u.")
    axs[1].set_ylabel("Relative change")
    axs[2].set_ylabel("Relative change")

    for ax in axs:
        ax.set_xlim(xlim_min, xlim_max)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    #set colorbar
    cax = fig.add_subplot(gs[:, 1])
    sm = cm.ScalarMappable(norm=cm_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label="Delay time /ps")

    fig.tight_layout()
    pdf.savefig(fig)
#########################################################################################
    #plot transients of peaks

    fig, axs = plt.subplots(4,3, figsize=figsize, sharex=True)
    fig.suptitle(f"Peak transients, normalized to first {pre_t0} ps")

    for ax in axs[-1]: #get last ax in each column
        ax.set_xlabel("Delay /ps")
    axs = axs.flatten()

    for i, (ax, roi) in enumerate(zip(axs, rois)):
        intgr = np.sum(los_norm[:, roi], axis=-1) / np.mean(np.sum(los_norm[:pre_idx, roi], axis=-1))
        
        if fit_flag:
            initial_guess = (2, 0, intgr.max()-intgr.min(), intgr.max()-intgr.min(), 0.6, 100, intgr.min()) #time_resolution, t0_fit, amp1, amp2, tconst1, tconst2, offset
            try:
                bounds = ([0,    -np.inf, -np.inf, -np.inf, 0,   0,   -np.inf],  # lower
                          [np.inf, np.inf,  np.inf,  np.inf, np.inf, np.inf,  np.inf])   # upper
                params, _ = curve_fit(irf_fit, delay_times, intgr, p0=initial_guess, bounds=bounds)
                time_resolution, t0_fit, amp1, amp2, tconst1, tconst2, offset = params
                best_fit_curve = irf_fit(delay_times, *params)
                ax.axvline(t0_fit,
                           linestyle="--",
                           color="grey",
                           linewidth=1,
                           zorder=-10,
                           label=f"t0={t0_fit:.1f}"
                )
                ax.plot(delay_times, best_fit_curve, color="silver", lw=3, 
                        label=r"$\{\tau\}=\{"
                            + f"{tconst1:.2f}, {tconst2:.0f}"
                            + r"\}$"
                            + f"\nirf={time_resolution:.2f}"
                            )
            except (RuntimeError, ValueError):    
                print(f"fit failed at q={qs[peaks[i]]:.2f} A^-1")

        ax.axvspan(delay_times[:pre_idx].min(), delay_times[:pre_idx].max(), alpha=0.3, color="silver", label="Used for norm.")
        ax.axhline(np.mean(intgr[:pre_idx]), color="r", label="average, 1??")
        ax.plot(delay_times, intgr, zorder=5, color="k")
        ax.axvline(0, color="silver")
        ax.scatter(delay_times, intgr, color=cmap(np.linspace(0, 1, len(delay_times))), zorder=10)
        ax.set_xlim(delay_times[0], delay_times[-1])

    fig.tight_layout()
    for ax in axs:
        if ax.get_legend_handles_labels() != ([], []):
            ax.legend(fontsize=8)
    pdf.savefig(fig)

#########################################################################################
    #plot all lineouts and differences over time




