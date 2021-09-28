import analyse_logic as al
import file_logic as file
import plot_logic as pl
import numpy as np
from matplotlib.pyplot import show


"Plot: Beispielschwingung"
def Plot_Beispielschwingung(savefigs=True):

    file_dicts = file.file_search(title='AbklingASaite', raw_data=True)

    file_number = 7
    file_dict = file_dicts[file_number]

    time_data, ampl_data = file.load_file(file_dict)

    # Plotting
    pl.setparams(savefigs)

    normal_fig_name = 'Beispielschwingung'
    time_cut = 21
    normal_fig, normal_ax = pl.right_subplots(width='half')
    normal_ax.plot(time_data[time_data < time_cut], ampl_data[time_data < time_cut], 'k', linewidth=0.1)
    normal_ax.set_xlim(right=time_cut)
    normal_ax.set_ylabel('Amplitude in a.u.')
    normal_ax.set_xlabel('Zeit in s')
    pl.siunitx_ticklabels(normal_ax, xaxis=False, y_precision=3)

    zoom_fig_name = 'Beispielschwingung_zoom'
    time_cut_bounds = (4.426, 4.524)
    time_cut = al.range_cut(time_data, *time_cut_bounds)
    zoom_fig, zoom_ax = pl.right_subplots(width='half')
    zoom_ax.plot(time_data[time_cut], ampl_data[time_cut], 'k')
    zoom_ax.set_xlim(*time_cut_bounds)
    zoom_ax.set_ylabel('Amplitude in a.u.')
    zoom_ax.set_xlabel('Zeit in s')
    pl.siunitx_ticklabels(zoom_ax, 'DE', True, 2, None, True, 4)
    if savefigs:
        pl.savefig(normal_fig_name, normal_fig, pl.AUSWERTUNG_DIR)
        pl.savefig(zoom_fig_name, zoom_fig, pl.AUSWERTUNG_DIR)
    else:
        show()


"Anregungsfrequenz"
"Plot: Mittelwert-Übersicht"
def Plot_Demod_Mittelwert(savefigs=True):
    slow_file_dict = file.file_search(title='demodSlowSweep', date='19-11-21', description='SNRData', raw_data=True)
    f, pl_mean = file.load_file(slow_file_dict, (0, 2))

    pl_mean = pl_mean / np.amax(pl_mean)

    pl.setparams(savefigs)

    fig, ax = pl.right_subplots()
    ax.plot(f*1e-6, pl_mean, '-k')
    ax.set_xlabel('Anregungsfrequenz in MHz')
    ax.set_ylabel("PL-Intensität normiert in a.u.")

    pl.siunitx_ticklabels(ax)
    if savefigs:
        pl.savefig('DemodMeanOverview', fig, fig_directory=pl.OPTIMIERUNG_DIR)
    else:
        show()

"Magnetpos"
"Plot: Sinus Absolut"
def Plot_Sinus_Absolut(savefigs=True):
    file_dict = file.file_search(title="TestMagnet")

    time_data, ampl_data = file.load_file(file_dict)
    sampling_rate = al.get_data_sampling_rate(time_data)

    # normieren
    ampl_data -= np.mean(ampl_data[time_data < 0.05])
    ampl_data /= np.amax(ampl_data)

    # Plotting
    pl.setparams(savefigs)

    time_cut = al.range_cut(time_data, 3.1)
    normal_fig, normal_ax = pl.right_subplots()
    normal_ax.plot(time_data[time_cut], ampl_data[time_cut], linewidth=0.3)
    normal_ax.set_xlim((-0.11, 3.1))
    pl.time_label(normal_ax)
    normal_ax.set_ylabel("Amplitude in a.u.")

    pl.siunitx_ticklabels(normal_ax)

    if savefigs:
        pl.savefig("SinusAbsolut_t", normal_fig, fig_directory=pl.OPTIMIERUNG_DIR)
    else:
        show()

"Plot: Magnetposition Vergleich"
def Plot_Magnetposition_Vergleich(savefigs=True, latex=True):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    bad_file_dict = file.file_search(title="TestMagnet")
    good_file_dict = file.file_search(title='AbklingBSaite', raw_data=True, version_index=2, measurement_index=6)

    if latex:
        pl.setparams(savefigs)
    figures = []

    norm_factor = 1
    for file_no, file_dict in enumerate([bad_file_dict, good_file_dict]):
        time_data, ampl_data = file.load_file(file_dict)
        sampling_rate = al.get_data_sampling_rate(time_data)

        # normieren
        ampl_data -= np.mean(ampl_data[time_data < 0.05])
        if file_no == 0:
            norm_factor = np.amax(ampl_data)
        ampl_data /= norm_factor

        if file_no == 0:
            normal_ax_ylim = -0.4
        else:
            normal_ax_ylim = np.amin(ampl_data[time_data < 4])


        # FFT
        time_cut = 30.1
        freq_data, fft_data = al.fft(time_data[time_data < time_cut], ampl_data[time_data < time_cut], sampling_rate)
        first_peak_height = np.amax(fft_data[al.range_cut(freq_data, 28, 33)])

        # Plotting
        normal_fig, normal_ax = pl.right_subplots(width=pl.WIDTH / 2)
        normal_ax.plot(time_data[time_data < time_cut], ampl_data[time_data < time_cut], linewidth=0.3)
        normal_ax.set_xlim(right=time_cut)
        normal_ax.set_ylim(bottom=1.04 * normal_ax_ylim)
        pl.time_label(normal_ax)
        normal_ax.set_ylabel("Amplitude in a.u.")
        pl.siunitx_ticklabels(ax=normal_ax, xaxis=False, y_precision=1)

        figures.append(normal_fig)

        fft_fig, fft_ax = pl.right_subplots(width=pl.WIDTH / 2)

        freq_cut_bounds = (18, 132)
        freq_cut = al.range_cut(freq_data, *freq_cut_bounds)
        fft_ax.plot(freq_data[freq_cut], fft_data[freq_cut], linewidth=0.7)
        fft_ax.set_xlim(freq_cut_bounds)
        pl.freq_label(fft_ax)
        fft_ax.set_ylabel('FFT-Amplitude in a.u.')
        pl.siunitx_ticklabels(fft_ax, xaxis=False, y_precision=2)

        x1, x2, y1, y2 = 28.2, 33.8, -0.001, 1.03 * first_peak_height       # specify the limits
        freq_cut_inset = al.range_cut(freq_data, x1, x2)
        fft_ax_inset = inset_axes(fft_ax, width="43%", height="50%")
        fft_ax_inset.plot(freq_data[freq_cut_inset], fft_data[freq_cut_inset], linewidth=0.5)
        fft_ax_inset.set_xlim(x1, x2)  # apply the x-limits
        # fft_ax_inset.set_ylim(y1, y2)  # apply the y-limits

        if file_dict == bad_file_dict:
            yticks = [0, 0, 0.05]
        else:
            yticks = [0, 0, 0.05, 0.1]
        xticks = [0, 30, 32.5]
        pl.siunitx_ticklabels(fft_ax_inset, xaxis=True, y_precision=2, yticks=yticks, xticks=xticks)
        # pl.siunitx_ticklabels(fft_ax_inset, y_precision=2)

        # pl.freq_label(fft_ax_inset)
        # fft_ax_inset.set_ylabel('FFT-Amplitude in a.u.')

        rect, lines = fft_ax.indicate_inset_zoom(fft_ax_inset)
        lines[0].set_visible(False)
        lines[1].set_visible(True)
        lines[2].set_visible(True)
        lines[3].set_visible(False)

        figures.append(fft_fig)

    figure_names = ["schlechteMagnetPosition_t", "schlechteMagnetPosition_f", "guteMagnetPosition_t", "guteMagnetPosition_f"]

    if savefigs:
        for fig_no in range(len(figure_names)):
            pl.savefig(figure_names[fig_no], figures[fig_no], fig_directory=pl.OPTIMIERUNG_DIR)
    else:
        show()


def Plot_Magnetposition_Vergleich_anders_Test(savefigs=False, latex=True):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Rectangle

    bad_file_dict = file.file_search(title="TestMagnet")
    good_file_dict = file.file_search(title='AbklingBSaite', raw_data=True, version_index=2, measurement_index=6)

    if latex:
        pl.setparams(savefigs)
    figures = []

    norm_factor = 1
    for file_no, file_dict in enumerate([bad_file_dict, good_file_dict]):
        time_data, ampl_data = file.load_file(file_dict)
        sampling_rate = al.get_data_sampling_rate(time_data)

        # normieren
        ampl_data -= np.mean(ampl_data[time_data < 0.05])
        if file_no == 0:
            norm_factor = np.amax(ampl_data)
        ampl_data /= norm_factor

        if file_no == 0:
            normal_ax_ylim = -0.2
        else:
            normal_ax_ylim = np.amin(ampl_data[time_data < 4])


        # FFT
        time_cut = 26
        freq_data, fft_data = al.fft(time_data[time_data < time_cut], ampl_data[time_data < time_cut], sampling_rate)
        first_peak_height = np.amax(fft_data[al.range_cut(freq_data, 28, 33)])
        print(first_peak_height)

        # Plotting
        normal_fig, normal_ax = pl.right_subplots()
        normal_ax.plot(time_data[time_data < time_cut], ampl_data[time_data < time_cut], linewidth=0.3)
        normal_ax.set_xlim(right=time_cut)
        normal_ax.set_ylim(bottom=1.04 * normal_ax_ylim)
        pl.time_label(normal_ax)
        normal_ax.set_ylabel("Amplitude in a.u.")
        pl.siunitx_ticklabels(ax=normal_ax, xaxis=False)

        if file_no == 0:
            time_cut_time = 1.05
            time_cut = al.range_cut(time_data, time_cut_time)
            normal_ax_inset = inset_axes(normal_ax, width="42%", height="51%")
            normal_ax_inset.plot(time_data[time_cut], ampl_data[time_cut], linewidth=0.3)
            normal_ax_inset.set_xlim(right=time_cut_time)
            pl.time_label(normal_ax_inset)
            normal_ax_inset.set_ylabel("Amplitude in a.u.")
            ticks = [0, 0, 0.5, 1.0]
            pl.siunitx_ticklabels(ax=normal_ax_inset, yticks=ticks, xticks=ticks)

            rect, lines = normal_ax.indicate_inset_zoom(normal_ax_inset)
            lines[0].set_visible(False)
            lines[1].set_visible(True)
            lines[2].set_visible(True)
            lines[3].set_visible(False)

        figures.append(normal_fig)

        fft_fig, fft_ax = pl.right_subplots(width=pl.WIDTH / 2)
        freq_cut_bounds = (18, 132)
        freq_cut = al.range_cut(freq_data, *freq_cut_bounds)
        fft_ax.plot(freq_data[freq_cut], fft_data[freq_cut], linewidth=0.7)
        fft_ax.set_xlim(freq_cut_bounds)
        pl.freq_label(fft_ax)
        fft_ax.set_ylabel('FFT-Amplitude in a.u.')
        pl.siunitx_ticklabels(fft_ax, xaxis=False, y_precision=2)

        x1, x2, y1, y2 = 28.2, 33.8, -0.004, 1.05 * first_peak_height       # specify the limits
        fft_ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, facecolor='none', edgecolor='0.5', alpha=0.5, zorder=4.99))

        fft_zoom_fig, fft_zoom_ax = pl.right_subplots(width=pl.WIDTH / 2)
        freq_cut_zoom = al.range_cut(freq_data, x1, x2)
        fft_zoom_ax.plot(freq_data[freq_cut_zoom], fft_data[freq_cut_zoom], linewidth=0.5)
        fft_zoom_ax.set_xlim(x1, x2)
        fft_zoom_ax.set_ylim(y1, y2)
        pl.freq_label(fft_zoom_ax)
        fft_zoom_ax.set_ylabel('FFT-Amplitude in a.u.')
        pl.siunitx_ticklabels(fft_zoom_ax, xaxis=False, y_precision=2)

        figures.append(fft_fig)
        figures.append(fft_zoom_fig)

    figure_names = ["Magnet_bad_t", "Magnet_bad_f", "Magnet_bad_f_zoom", "Magnet_good_t", "Magnet_good_f", "Magnet_good_f_zoom"]

    if savefigs:
        for fig_no in range(len(figure_names)):
            pl.savefig(figure_names[fig_no], figures[fig_no], fig_directory=pl.OPTIMIERUNG_DIR)
    else:
        show()

"Plot: zDipASaite"
def Plot_zDipASaite(savefigs=True):

    file_dict = file.file_search(title='zDipASaite', raw_data=False)

    z_data, pl_mean_data = file.load_file(file_dict, axis=None)


    pl_mean_data = pl_mean_data[z_data > 14]
    z_data = z_data[z_data > 14]
    z_data -= z_data[0]

    pl_mean_data -= np.amax(pl_mean_data)
    pl_mean_data /= -np.amin(pl_mean_data)

    pl.setparams(savefigs)
    fig, ax = pl.right_subplots()
    ax.plot(z_data, pl_mean_data, '.')
    ax.set_xlabel(r'$z$ in mm')
    ax.set_ylabel('PL normiert in a.u.')
    pl.siunitx_ticklabels(ax)
    if savefigs:
        pl.savefig('zDipASaite', fig, pl.OPTIMIERUNG_DIR)
    else:
        show()

"Plot: zDipBSaite"
# file_dict = file.file_search(title='zDipBSaite', raw_data=False)
#
# z_data, pl_mean_data = file.load_file(file_dict, axis=None)
#
# pl.setparams()
# fig, ax = pl.right_subplots()
# ax.plot(z_data, pl_mean_data, 'k')
# ax.set_xlabel('z in mm')
# ax.set_ylabel('PL in bel.Einh.')
#
# pl.savefig('zDipBSaite', fig)
# # show()


"Abklingverhalten"
"Datenvorbereitung"
"Filter Data A-Saite"
# al.filter_and_save((40,1000), 'band', title='AbklingASaite')
"Cut Data A-Saite"
# al.cut_and_save(title='AbklingASaite', filtering='high')
"oder Filter and Cut Data A-Saite"
# al.filter_cut_and_save((40, 1000), 'band', title='AbklingASaite')
"Cut Data B-Saite"
# al.cut_and_save(title='AbklingBSaite', filtering='band')
"Filter and Cut Data B-Saite"
# al.filter_cut_and_save((20, 1000), 'band', title='AbklingBSaite')

"Plot: vorher_nachher Filterung"
def Plot_vorher_nachher_Filterung(savefigs=True):
    from brokenaxes import brokenaxes

    unfilt_dicts = file.file_search(title="AbklingASaite", raw_data=True)
    filt_dicts = file.file_search(title="AbklingASaite", raw_data=False, filtering='band', is_cut=False)

    file_no = 3
    unfilt_dict = unfilt_dicts[file_no]
    # filt_dict = filt_dicts[file_no]

    unfilt_data = file.load_file(unfilt_dict)
    # filt_data = file.load_file(filt_dict)
    filt_data = unfilt_data.copy()
    filt_data[1] = al.data_filter(unfilt_data[1], (40, 1000), al.get_data_sampling_rate(unfilt_data[0]), 'band')

    time_cut = 10
    freq_vec, unfilt_fft = al.fft(*unfilt_data[:, unfilt_data[0] < time_cut])
    filt_fft = al.fft(*filt_data[:, filt_data[0] < time_cut])[1]

    freq_cut = 3200
    unfilt_fft = unfilt_fft[freq_vec < freq_cut]
    filt_fft = filt_fft[freq_vec < freq_cut]
    freq_vec = freq_vec[freq_vec < freq_cut]


    # Plotting t
    pl.setparams(savefigs)
    fig_dir = pl.ABKLING_DIR
    unfilt_color = '#1f77b4'
    filt_color = '#ff7f0e'

    normal_fig_name = 'VorherNachherFilter_Schw'
    normal_fig = pl.right_subplots(width='half', return_only_fig=True)
    plt_ylims = ((-0.0025828195479650493, 0.0015968800664547477), (1.3211427518818704, 1.3257845903734102))
    bax = brokenaxes(fig=normal_fig, ylims=plt_ylims, despine=False, hspace=.05)
    bax.plot(*unfilt_data, c=unfilt_color)
    bax.plot(*filt_data, c=filt_color)
    pl.time_label(bax.axs[1])
    bax.set_ylabel('Amplitude in a.u.')
    for bax_ax in bax.axs:
        pl.siunitx_ticklabels(bax_ax, xaxis=False, y_precision=3)

    fft_fig_name = 'VorherNachherFilter_FFT'
    fft_fig, fft_ax = pl.right_subplots(width='half')
    fft_ax.plot(freq_vec, unfilt_fft, c=unfilt_color)
    fft_ax.plot(freq_vec, filt_fft, c=filt_color)
    fft_ax.set_yscale('log')
    fft_ax.set_xlim(right=freq_cut)
    pl.freq_label(fft_ax)
    fft_ax.set_ylabel('FFT-Amplitude in a.u.')

    if savefigs:
        pl.savefig(normal_fig_name, normal_fig, fig_dir)
        pl.savefig(fft_fig_name, fft_fig, fig_dir)
    else:
        show()

"STFT"
"Plot: A-Saite STFT 3D-Beispiel"
def Plot_ASaite_STFT_Beispiel(savefigs=True):
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    file_dicts = file.file_search(title='AbklingASaite', raw_data=False, is_cut=True, filtering='band')

    file_number = 3
    file_dict = file_dicts[file_number]

    time_data, ampl_data = file.load_file(file_dict)

    sampling_rate = al.get_data_sampling_rate(time_data)

    nperseg = 2 ** 12 + 2 ** 11
    ngap = 2 ** 6 + 2 ** 5

    f, t, Zxx = al.stft_func(ampl_data, sampling_rate, nperseg=nperseg, ngap=ngap)
    freq_range = al.range_cut(f, 0, 1400)
    Zxx = Zxx[freq_range]
    f = f[freq_range]

    pl.setparams(savefigs)

    fig, ax = pl.right_subplots(width='full')
    im = ax.pcolormesh(t, f, Zxx, norm=LogNorm(), rasterized=True)
    cbar = fig.colorbar(im, ax=ax)
    ax.set_xlabel('Zeit in s')
    ax.set_ylabel('Frequenz in Hz')
    cbar.set_label('FT-Amplitude in a.u.')

    inset_ax = inset_axes(ax, '50%', '50%')
    inset_ax.pcolormesh(t, f, Zxx, norm=LogNorm(), rasterized=True)
    inset_ax.set_ylim((40, 240))
    inset_ax.set_xlim(right=30)
    inset_ax.set_xlabel('Zeit in s')
    inset_ax.set_ylabel('Frequenz in Hz')

    rect, lines = ax.indicate_inset_zoom(inset_ax)
    lines[0].set_visible(False)
    lines[1].set_visible(True)
    lines[2].set_visible(True)
    lines[3].set_visible(False)

    if savefigs:
        pl.savefig('ASaite_STFTBeispiel', fig, fig_directory=pl.ABKLING_DIR, dpi=400)
    else:
        show()

def Plot_Hamm_Window(savefigs=True):
    from scipy.signal import windows
    nperseg = 2 ** 12 + 2 ** 11
    window = windows.hamming(nperseg)

    pl.setparams(savefigs)

    fig, ax = pl.right_subplots()
    ax.plot(window)
    ax.set_ylabel('Wichtung')
    ax.set_xlabel('Datenpunkte $n$')
    ax.set_xlim((0, nperseg))
    ax.set_xticks([0, nperseg])
    ax.set_xticklabels([0, "$ N - 1 $"])
    ax.set_ylim(bottom=0)

    pl.siunitx_ticklabels(ax, xaxis=False)

    if savefigs:
        pl.savefig('Hamm_Fenster', fig, pl.ABKLING_DIR)
    else:
        show()

"Harmonische"
"A-Saite Harm-Speicherung"
# al.harm_save(55, title='AbklingASaite', description=None, raw_data=False, is_cut=True, filtering='band')
"B-Saite Harm-Speicherung"
# al.harm_save(31, title='AbklingBSaite', description=None, raw_data=False, is_cut=True, filtering='band')

"Plot:A-Saite Harm-Beispiel"

"Plot: A-Saite Harmonische"
def Plot_ASaite_Harmonische(savefigs=True):
    from matplotlib.colors import TABLEAU_COLORS

    time_data, harm_data_list, harm_data_mean, harm_data_std, harm_data_mean_sum, harm_data_std_sum \
        = al.Harm_mean_std_and_sum(title='AbklingASaite', description='harm', raw_data=False, is_cut=True, filtering='band')

    # Fitting
    c_mean_cut = 80
    fit_range = al.range_cut(time_data, 10, 40)

    # erste harm fitten
    erste_harm, erste_harm_popt = al.harm_fit_and_shift(time_data, harm_data_mean[0], c_mean_cut, fit_range)

    # sum fitten
    sum_harm, sum_harm_popt = al.harm_fit_and_shift(time_data, harm_data_mean_sum, c_mean_cut, fit_range)

    # Plotting
    # Plotparameter einstellen
    pl.setparams(savefigs)
    colors = TABLEAU_COLORS
    color_names = list(colors)
    fig_dir = pl.ABKLING_DIR

    # alle Daten auf Plotbereich beschneiden
    time_cut = (-.5, 50.8)
    time_range = al.range_cut(time_data, *time_cut)
    time_data = time_data[time_range]
    harm_data_mean = harm_data_mean[:, time_range]
    harm_data_std = harm_data_std[:, time_range]
    harm_data_mean_sum = harm_data_mean_sum[time_range]
    harm_data_std_sum = harm_data_std_sum[time_range]
    erste_harm = erste_harm[time_range]
    sum_harm = sum_harm[time_range]
    fit_range = fit_range[time_range]

    def Harm_mit_Fehler(save=True):
        fig_name = 'ASaite_Harmonische_mit_Fehlerschatten'
        fig, ax = pl.right_subplots()
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)

        ax.fill_between(time_data, harm_data_mean_sum-harm_data_std_sum, harm_data_mean_sum+harm_data_std_sum,
                        color='k', alpha=0.2)
        ax.plot(time_data, harm_data_mean_sum, label='Summe',
                color='k', linestyle='solid', linewidth=0.5, marker=None)

        for idx in range(6):
            ax.plot(time_data, harm_data_mean[idx], label="{}. Harmonische".format(idx+1),
                    color=colors[color_names[idx]], linestyle='solid', linewidth=0.5, marker=None)
            ax.fill_between(time_data,
                            harm_data_mean[idx]-harm_data_std[idx], harm_data_mean[idx]+harm_data_std[idx],
                            color=colors[color_names[idx]], alpha=0.2)

        pl.legend()
        pl.siunitx_ticklabels(ax, xaxis=False)

        if save:
            pl.savefig(fig_name, fig, fig_dir)

    def Harm_Log(save=True):
        fig_name = 'ASaite_Harmonische_log'
        fig, ax = pl.right_subplots()
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)
        ax.set_yscale('log')

        ax.plot(time_data, harm_data_mean_sum, label='Summe',
                color='k', linestyle='solid', linewidth=0.5, marker=None)

        for idx in range(6):
            ax.plot(time_data, harm_data_mean[idx], label="{}. Harmonische".format(idx+1),
                    color=colors[color_names[idx]], linestyle='solid', linewidth=0.5, marker=None)

        # ax.legend(ncol=2)

        if save:
            pl.savefig(fig_name, fig, fig_dir)

    def n_Harm_mit_Fit(n_harm_data, n_harm_popt, n=1, save=True):
        alpha = "{:.5f}".format(n_harm_popt[1])
        print("Alpha_", n, " = ", alpha)

        fig_name = 'ASaite_{}te_Harmonische_mit_Fit'.format(n)
        fig, ax = pl.right_subplots(width='half')
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)
        ax.set_yscale('log')

        ax.plot(time_data[:], n_harm_data[:], color=colors[color_names[n-1]], linewidth=0.5, label='{}. Harmonische'.format(n))
        ax.plot(time_data[:], al.exp_func(time_data[:], *n_harm_popt), 'r--', linewidth=0.5,
                # label=r'exponentieller Fit mit $\alpha = \SI{{{0:4f}}}{{\second^{{-1}}}}$'.format(n_harm_popt[1]))
                label=r'exp. Fit mit $\alpha = \SI{{{0}}}{{\second^{{-1}}}}$'.format(alpha))

        pl.legend(loc='lower left')

        if save:
            pl.savefig(fig_name, fig, fig_dir)

    def Summe_Harmonische_mit_Fit(save=True):
        alpha = "{:.5f}".format(sum_harm_popt[1])
        print("Alpha_Summe = ", alpha)

        fig_name = 'ASaite_SummeHarmonische_mit_Fit'
        fig, ax = pl.right_subplots(width='half')
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)
        ax.set_yscale('log')

        ax.plot(time_data, sum_harm, 'k', linewidth=0.5, label='Summe')
        ax.plot(time_data, al.exp_func(time_data, *sum_harm_popt),
                'r--', linewidth=0.5,
                # label=r'exponentieller Fit mit $\alpha = \SI{{{0:4f}}}{{\second^{{-1}}}}$'.format(sum_harm_popt[1]))
                label=r'exp. Fit mit $\alpha = \SI{{{}}}{{\second^{{-1}}}}$'.format(alpha))

        pl.legend(loc='lower left')
        if save:
            pl.savefig(fig_name, fig, fig_dir)

    Harm_mit_Fehler(savefigs)
    Harm_Log(savefigs)
    n_Harm_mit_Fit(erste_harm, erste_harm_popt, save=savefigs)
    Summe_Harmonische_mit_Fit(savefigs)
    if not savefigs:
        show()


"Plot: B-Saite Harmonische"
def Plot_BSaite_Harmonische(savefigs=True):
    from matplotlib.colors import TABLEAU_COLORS

    time_data, harm_data_list, harm_data_mean, harm_data_std, harm_data_mean_sum, harm_data_std_sum \
        = al.Harm_mean_std_and_sum(8, title='AbklingBSaite', date='19-12-18',
                                   description='harm', raw_data=False, is_cut=True, filtering='band')

    # Fitting
    c_mean_cut = 100
    fit_range = al.range_cut(time_data, 10, 50)

    # erste harm fitten
    zweite_harm, zweite_harm_popt = al.harm_fit_and_shift(time_data, harm_data_mean[1], c_mean_cut, fit_range)

    # dritte harm fitten
    dritte_harm, dritte_harm_popt = al.harm_fit_and_shift(time_data, harm_data_mean[2], c_mean_cut, fit_range)

    # sum fitten
    fit_range_for_sum = al.range_cut(time_data, 0.4, 50)
    sum_harm, sum_harm_popt = al.harm_fit_and_shift(time_data, harm_data_mean_sum, c_mean_cut, fit_range_for_sum)

    # Plotting
    # Plotparameter einstellen
    pl.setparams(savefigs)
    colors = TABLEAU_COLORS
    color_names = list(colors)
    fig_dir = pl.ABKLING_DIR

    # alle Daten auf Plotbereich beschneiden
    time_cut = (-.5, 50.8)
    time_range = al.range_cut(time_data, *time_cut)
    time_data = time_data[time_range]
    harm_data_mean = harm_data_mean[:, time_range]
    harm_data_std = harm_data_std[:, time_range]
    harm_data_mean_sum = harm_data_mean_sum[time_range]
    harm_data_std_sum = harm_data_std_sum[time_range]
    zweite_harm = zweite_harm[time_range]
    dritte_harm = dritte_harm[time_range]
    sum_harm = sum_harm[time_range]
    fit_range = fit_range[time_range]

    # Plot Funktionen
    def Harm_mit_Fehler(save=True):
        fig_name = 'BSaite_Harmonische_mit_Fehlerschatten'
        fig, ax = pl.right_subplots()
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)

        ax.fill_between(time_data, harm_data_mean_sum-harm_data_std_sum, harm_data_mean_sum+harm_data_std_sum,
                        color='k', alpha=0.2)
        ax.plot(time_data, harm_data_mean_sum, label='Summe',
                color='k', linestyle='solid', linewidth=0.5, marker=None)

        for idx in range(6):
            ax.plot(time_data, harm_data_mean[idx], label="{}. Harmonische".format(idx+1),
                    color=colors[color_names[idx]], linestyle='solid', linewidth=0.5, marker=None)
            ax.fill_between(time_data,
                            harm_data_mean[idx]-harm_data_std[idx], harm_data_mean[idx]+harm_data_std[idx],
                            color=colors[color_names[idx]], alpha=0.2)

        pl.legend()
        pl.siunitx_ticklabels(ax, xaxis=False)
        if save:
            pl.savefig(fig_name, fig, fig_dir)

    def Harm_Log(save=True):
        fig_name = 'BSaite_Harmonische_log'
        fig, ax = pl.right_subplots()
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)
        ax.set_yscale('log')

        ax.plot(time_data, harm_data_mean_sum, label='Summe',
                color='k', linestyle='solid', linewidth=0.5, marker=None)

        for idx in range(6):
            ax.plot(time_data, harm_data_mean[idx], label="{}. Harmonische".format(idx+1),
                    color=colors[color_names[idx]], linestyle='solid', linewidth=0.5, marker=None)

        # ax.legend(ncol=2)

        if save:
            pl.savefig(fig_name, fig, fig_dir)

    def n_Harm_mit_Fit(n_harm_data, n_harm_popt, n=1, save=True):
        alpha = "{:.5f}".format(n_harm_popt[1])
        print("Alpha_", n, " = ", alpha)

        fig_name = 'BSaite_{}te_Harmonische_mit_Fit'.format(n)
        fig, ax = pl.right_subplots(width='half')
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)
        ax.set_yscale('log')

        ax.plot(time_data[:], n_harm_data[:], color=colors[color_names[n-1]], linewidth=0.5, label='{}. Harmonische'.format(n))
        ax.plot(time_data[:], al.exp_func(time_data[:], *n_harm_popt), 'r--', linewidth=0.5,
                label=r'exp. Fit mit $\alpha = \SI{{{}}}{{\second^{{-1}}}}$'.format(alpha))
        pl.legend()
        if save:
            pl.savefig(fig_name, fig, fig_dir)

    def Summe_Harmonische_mit_Fit(save=True):
        alpha = "{:.5f}".format(sum_harm_popt[1])
        print("Alpha_Summe = ", alpha)

        fig_name = 'BSaite_SummeHarmonische_mit_Fit'
        fig, ax = pl.right_subplots(width=0.5 * pl.WIDTH / pl.GOLDEN)
        ax.set_xlabel('Zeit in s')
        ax.set_ylabel('FT-Amplitude normiert in a.u.')
        ax.set_xlim(*time_cut)
        ax.set_yscale('log')

        ax.plot(time_data, sum_harm, 'k', linewidth=0.5, label='Summe')
        ax.plot(time_data, al.exp_func(time_data, *sum_harm_popt),
                'r--', linewidth=0.5,
                label=r'exp. Fit mit $\alpha = \SI{{{}}}{{\second^{{-1}}}}$'.format(alpha))
        pl.legend()
        if save:
            pl.savefig(fig_name, fig, fig_dir)

    # Ausführen der Funktionen
    Harm_mit_Fehler(savefigs)
    Harm_Log(savefigs)
    n_Harm_mit_Fit(zweite_harm, zweite_harm_popt, n=2, save=savefigs)
    n_Harm_mit_Fit(dritte_harm, dritte_harm_popt, n=3, save=savefigs)
    Summe_Harmonische_mit_Fit(savefigs)
    if not savefigs:
        show()

# 1. harm anfitten (bereich gut wählen)
# 1.-5.(6.) ohne fehler log plot
# summe, 1. - 4. lin plot mit fehler
# störungen erklären (schwebung durch gestell, Position nicht gut, energieübertragung(?)


###########################
'Ausführen der Funktionen'
'#########################'

# Plot_Beispielschwingung(True)
# Plot_Demod_Mittelwert(True)
# Plot_Magnetposition_Vergleich_anders_Test(True)

# Plot_zDipASaite(True)
# Plot_vorher_nachher_Filterung(True)
# Plot_ASaite_STFT_Beispiel(True)
# Plot_Hamm_Window(True)

Plot_ASaite_Harmonische(True)
Plot_BSaite_Harmonische(True)

