import numpy as np
from uedhhlib.fitting import pvoigt_2d, fit_pvoigt_2d
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colorbar
import matplotlib.patches as patches
from pathlib import Path
import h5py
from tqdm import tqdm

analysis_folder = Path(__file__).parent

hdf5_file = Path(r"Z:\Users\Emma\Zyla1\2026\2026_08\NT53\260807\Pos2A_TcF4\Meas1_filtered\hlh_analysis\exp1_woarcs_2ndIter.h5")

window_size = 3 #for convolve
timezero = 6 #in ps

### list of peaks to fit. Add center coordinates as tuple (x,y)
peaks_A = [(1677, 865), (1855, 821), (1633, 700), (1726, 1211), (1641, 1231), (1438, 567)]
peaks_B = [(1322, 958), (1142, 1007), (1367, 1123), (1263, 619), (1356, 590), (1555, 1258)]
peaks =[]
peaks.extend(peaks_A+peaks_B)
###todo: choose delaysteps (in peakpixelpos_lib? as function with if statement?)
# delta_t_min = 0
# delta_t_max = 
###
PADDING = 25 
FIGSIZE_SMALL = (16,9)
FIGSIZE_LARGE = (18,9)    


def colors_from_arr(a, cmap='viridis', start=0, end=1, log_scale=False):
    cmap = plt.get_cmap(cmap)
    a = np.array(a)
    a = a.astype("float")
    a += np.abs(a.min())
    a -= a.min()
    if log_scale:
        a = np.log1p(a)
    a /= a.max()
    color_scale = start + a * (end - start)
    return cmap(start + a * (end - start))

with h5py.File(hdf5_file) as h:
    data = h["processed/equilibrium"][()]
    data_time = h["processed/intensity"][()]
    time_points = h["time_points"][()]

data_time = np.moveaxis(data_time, -1, 0)


with PdfPages(analysis_folder /f"peakfit_exp1_{len(peaks_A)+len(peaks_B)}pks.pdf") as pdf:
    ### overview page
    f, axs = plt.subplots(1,2, figsize=FIGSIZE_SMALL)
    f.suptitle("Overview Bragg Reflection", fontsize=16)
    axs[0].imshow(data, cmap = "bone_r", vmin=1, vmax=np.percentile(data,99.75), aspect="equal")
    axs[0].set_title("linear scale")
    axs[1].imshow(data, cmap = "bone_r", vmin=1, vmax=np.percentile(data,99.75), aspect="equal", norm="log")
    axs[1].set_title("log scale")

    #for i, peak in enumerate(peaks):
    for i, (peak_A, peak_B) in enumerate(zip(peaks_A, peaks_B)):
        for ax in axs:
            for peak, label in zip([peak_A, peak_B], [f"{i+1}_A", f"{i+1}_B"]):
                ax.add_patch(
                patches.Rectangle(
                (peak[0] - PADDING, peak[1] - PADDING),
                PADDING * 2,
                PADDING * 2,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                ))

                ax.text(peak[0], peak[1], label, color="r", fontsize=10)
    
    f.tight_layout()
    pdf.savefig(f)


    amp_list_A = []
    amp_list_B = []
    amp_err_list_A = []
    amp_err_list_B = []

    for i, (peak_A, peak_B) in tqdm(enumerate(zip(peaks_A, peaks_B)), desc="Fitting peak pairs"): #iterate over pairs
        ### static fitting
        bragg_data_A = data_time[0,
            peak_A[1] - PADDING : peak_A[1] + PADDING,
            peak_A[0] - PADDING : peak_A[0] + PADDING,
        ]

        bragg_data_B = data_time[0,
            peak_B[1] - PADDING : peak_B[1] + PADDING,
            peak_B[0] - PADDING : peak_B[0] + PADDING,
        ]

        res_A = fit_pvoigt_2d(bragg_data_A)
        res_B = fit_pvoigt_2d(bragg_data_B)

        ### colorplots of peaks, fits and residuum
        f, axs = plt.subplots(2,4, figsize=FIGSIZE_SMALL, gridspec_kw={"width_ratios":[0.32,0.32,0.32,0.04]})
        f.suptitle(f"Friedel pair {i+1} Fit Cverview", fontsize=16)

        axs = axs.flatten()

        axs[0].imshow(bragg_data_A, cmap = "inferno", vmin=0, vmax=np.percentile(bragg_data_A,99.75), aspect="equal")
        axs[0].set_title("data_A")

        axs[1].imshow(res_A.best_fit.reshape(2*PADDING,2*PADDING) ,cmap = "inferno", vmin=0, vmax=np.percentile(bragg_data_A,99.71), aspect="equal")
        axs[1].set_title("best fit_A")

        residuals_best_A = axs[2].imshow(bragg_data_A-res_A.best_fit.reshape(2*PADDING,2*PADDING), cmap="bwr", aspect="equal", vmin=-2, vmax=2)
        axs[2].set_title("residuals (data_A - best fit_A)")

        f.colorbar(residuals_best_A, cax=axs[3], label="$residuum$ /cts")

        axs[4].imshow(bragg_data_B, cmap = "inferno", vmin=0, vmax=np.percentile(bragg_data_B,99.75), aspect="equal")
        axs[4].set_title("data_B")

        axs[5].imshow(res_B.best_fit.reshape(2*PADDING,2*PADDING) ,cmap = "inferno", vmin=0, vmax=np.percentile(bragg_data_B,99.71), aspect="equal")
        axs[5].set_title("best fit_B")

        residuals_best_B = axs[6].imshow(bragg_data_B-res_B.best_fit.reshape(2*PADDING,2*PADDING), cmap="bwr", aspect="equal", vmin=-2, vmax=2)
        axs[6].set_title("residuals (data_B - best fit_B)")

        f.colorbar(residuals_best_B, cax=axs[7], label="$residuum$ /cts")

        f.tight_layout()
        pdf.savefig(f)

        ### time series fitting
        m_A, amp_A, amp_err_A, x0_A, y0_A, bg_offset_A = [],[],[],[],[],[]
        m_B, amp_B, amp_err_B, x0_B, y0_B, bg_offset_B = [],[],[],[],[],[]

        initial_guess_A = [res_A.params[name].value for name in res_A.var_names]
        initial_guess_B = [res_B.params[name].value for name in res_B.var_names]

        for t, d in zip(time_points, data_time[:]):
            bragg_data_A = d[
            peak_A[1] - PADDING : peak_A[1] + PADDING,
            peak_A[0] - PADDING : peak_A[0] + PADDING,
        ]
            bragg_data_B = d[
            peak_B[1] - PADDING : peak_B[1] + PADDING,
            peak_B[0] - PADDING : peak_B[0] + PADDING,
        ]


            temp_res_A = fit_pvoigt_2d(bragg_data_A, initial_guess=initial_guess_A)
            temp_res_B = fit_pvoigt_2d(bragg_data_B, initial_guess=initial_guess_B)
        
            for name, value_list_A, value_list_B in zip(['m', 'amp', 'x0', 'y0', 'bg_offset'], 
                                                        [m_A, amp_A, x0_A, y0_A, bg_offset_A],
                                                        [m_B, amp_B, x0_B, y0_B, bg_offset_B]):
                value_list_A.append(temp_res_A.values[name])
                value_list_B.append(temp_res_B.values[name])

            amp_err_A.append(temp_res_A.params["amp"].stderr)
            amp_err_B.append(temp_res_B.params["amp"].stderr)

        amp_list_A.append(amp_A)
        amp_list_B.append(amp_B)
        amp_err_list_A.append(amp_err_A) 
        amp_err_list_B.append(amp_err_B) 

        ### plots of peak movement and mixing ratio
        f, axs = plt.subplots(2,3, figsize=FIGSIZE_SMALL, sharex=True)
        f.suptitle(f"Friedel pair {i+1} movement and shape")

        for ax, val, name, title in zip(axs.flatten(), 
                                 [x0_A, y0_A, m_A, x0_B, y0_B, m_B], 
                                 ['center x0_A', 'center y0_A', 'mixing ratio_A', 'center x0_B', 'center y0_B', 'mixing ratio_B'],
                                 ["_A", "_A", "_A", "_B", "_B", "_B"]):
            ax.scatter(time_points[:], val[:],c=colors_from_arr(time_points[:], cmap="plasma"))
            ax.set_xlabel("$delay time$ /ps")
            ax.set_ylabel(f"{name}")
            ax.set_title(f"{title}")
            ax.grid()

        f.tight_layout()
        pdf.savefig(f)

    
        amp_ave = (np.array(amp_A) + np.array(amp_B))/2
        amp_err_ave = np.sqrt(np.array(amp_err_A)**2 + np.array(amp_err_B)**2) / 2

        norm_amp_ave = (amp_ave/np.mean(amp_ave[:5]))*100
        norm_amp_err_ave = np.sqrt((100*window_size/np.sum(amp_ave[:5]))**2 * amp_err_ave**2 +(window_size*100*amp_ave/np.sum(amp_ave[:5])**2)**2*(amp_err_ave[0]**2+amp_err_ave[1]**2+amp_err_ave[2]**2+amp_err_ave[3]**2+amp_err_ave[4]**2))

        kernel = np.ones(window_size)/window_size
        smoothed_time_points = np.convolve(time_points[:], kernel, mode='valid')
        smoothed_amp_ave = np.convolve(norm_amp_ave[:], kernel, mode='valid')
        smoothed_amp_err_ave = np.sqrt(np.convolve(np.array(norm_amp_err_ave)[:]**2, np.ones(window_size), mode='valid'))/window_size



        ### plot averaged amplitude of the friedel pair with error bars
        
        time_points_corr = np.array(time_points) - timezero
        smoothed_time_points_corr = np.array(smoothed_time_points) - timezero
        
        f, ax = plt.subplots(1,1, figsize=FIGSIZE_SMALL)
        colors = colors_from_arr(time_points_corr, cmap="plasma", start=0, end=0.95, log_scale=True)
        ax.plot(smoothed_time_points_corr, smoothed_amp_ave, c="silver", zorder=1)
        ax.scatter(time_points_corr, norm_amp_ave, c=colors, zorder=2)
        for j, (x,y,yerr,color) in enumerate(zip(time_points_corr, norm_amp_ave, norm_amp_err_ave, colors)):
            ax.errorbar(x, y, yerr=yerr, color=color)
        #ax.set_xscale('log')
        ax.set_xlim(time_points_corr[0], max(time_points_corr))
        ax.set_xlabel("$Time$ /ps", fontsize=16)
        ax.set_ylabel("$Relative$ $intensity$ /%", fontsize=16) 
        ax.set_title(f"average amplitude of Friedel pair {i+1}", fontsize=16)
        ax.grid()
    
        f.tight_layout()
        pdf.savefig(f)

    #########################################################################################

        f, ax = plt.subplots(1,1, figsize=FIGSIZE_SMALL)
        colors = colors_from_arr(time_points_corr, cmap="plasma", start=0, end=0.95, log_scale=True)
        ax.plot(smoothed_time_points_corr, smoothed_amp_ave, c="silver", zorder=1)
        ax.scatter(time_points_corr, norm_amp_ave, c=colors, zorder=2)
        for j, (x,y,yerr,color) in enumerate(zip(time_points_corr, norm_amp_ave, norm_amp_err_ave, colors)):
            ax.errorbar(x, y, yerr=yerr, color=color)
        #ax.set_xscale('log')
        ax.set_xlim(-10, 10)
        ax.set_xlabel("$Time$ /ps", fontsize=16)
        ax.set_ylabel("$Relative$ $intensity$ /%", fontsize=16) 
        ax.set_title(f"average amplitude of Friedel pair {i+1} - zoom", fontsize=16)
        ax.grid()

    
        f.tight_layout()
        pdf.savefig(f)

        # y_ft = np.fft.fft(norm_amp_ave[6:30])
        # freqs = np.fft.fftfreq(len(norm_amp_ave[6:30]))#, d=(time_points[1]-time_points[0]))

        # plt.plot(freqs, np.abs(y_ft))  # Betrag der FFT
        # plt.xlabel("Frequenz")
        # plt.ylabel("Amplitude")
        # plt.show()