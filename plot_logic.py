import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

FIGURE_DIRECTORY = '\\Graphen\\'
AUSWERTUNG_DIR = os.path.join(FIGURE_DIRECTORY, "Auswertung\\")
ABKLING_DIR = os.path.join(FIGURE_DIRECTORY, "Auswertung\\Abkling\\")
OPTIMIERUNG_DIR = os.path.join(FIGURE_DIRECTORY, "Auswertung\\Optimierung\\")
WIDTH = 5.1
GOLDEN = (np.sqrt(5.) - 1) / 2.

params = {'text.usetex': True,
          'text.latex.preamble': r"\usepackage{lmodern} "
                                 r"\usepackage{siunitx} "
                                 r"\sisetup{exponent-product=\cdot, output-decimal-marker={,}, group-digits=integer, separate-uncertainty=true, multi-part-units=single}",
          'font.size': 11,
          'font.family': 'lmodern',
          'savefig.format': 'pdf',
          'savefig.dpi': 1200,
          'savefig.bbox': 'tight',
          'xtick.labelsize': 8,  # 6
          'ytick.labelsize': 8,  # 6
          'axes.labelsize': 10,  # 8
          'legend.fontsize': 8,  # 6
          'legend.markerscale': 4,
          # 'lines.marker': '.',
          # 'lines.linestyle': 'none',
          'axes.prop_cycle': mpl.cycler(color=['k']),
          'lines.markeredgecolor': 'k',
          'lines.markerfacecolor': 'k',
          'lines.markersize': 4,
          'lines.linewidth': 0.8  # 0.5
          }


def setparams(use_backend=True):
    if use_backend:
        mpl.use('pdf')

    plt.rcParams.update(params)


def nice_plots(show=False, legend=True):
    plt.get_current_fig_manager().window.showMaximized()
    if legend:
        leg = plt.legend()
        plt.setp(leg.get_lines(), linewidth=3)
    if show:
        plt.show()


def nice_legend(ax=None, linewidth=3.5):
    if ax is None:
        leg = plt.legend()
    else:
        leg = ax.legend()
    plt.setp(leg.get_lines(), linewidth=linewidth)


def nice_subplots(nrows=1, ncols=1, tight=True, maximized=True, **kwargs):
    kwargs['tight_layout'] = tight
    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    if maximized:
        plt.get_current_fig_manager().window.showMaximized()
    return fig, axs


def nax():
    return nice_subplots()[1]


def right_subplots(nrows=1, ncols=1, width=0.85*WIDTH, height=None, return_only_fig=False, **kwargs):
    if width is 'half':
        width = WIDTH / 2.
        if height is None:
            height = width
    elif width is 'full':
        width = WIDTH
    if height is None:
        height = width * GOLDEN * nrows / ncols

    if return_only_fig:
        return plt.figure(figsize=(width, height), **kwargs)
    else:
        return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def savefig(fig_name, fig, fig_directory=FIGURE_DIRECTORY, **savefig_kwargs):
    path_to_fig = os.path.join(fig_directory, fig_name)
    os.makedirs(fig_directory, exist_ok=True)

    fig.savefig(path_to_fig, **savefig_kwargs)


def time_label(labeled_ax):
    labeled_ax.set_xlabel("Zeit in s")


def freq_label(labeled_ax, prefix=None):
    if prefix is not None:
        label = "Frequenz in {}Hz".format(prefix)
    else:
        label = "Frequenz in Hz"
    labeled_ax.set_xlabel(label)


def siunitx_ticklabels(ax=None, locale="DE",
                       xaxis=True, x_precision=1, xticks=None,
                       yaxis=True, y_precision=1, yticks=None):
    """
    This function uses siunitx to create the ticklabels
    Main reason is for adjusting the decimal marker properly.
    The function takes 4 arguments:
        ax=None     the matplotlib axes to operate on
                    if set to None (Standard) this will be the current axes
        locale="DE" The locale parameter for siunitx, one of
                    "UK", "US", "DE", "FR" oder "ZA"
        xaxis=True  Boolean, if True the labels for the xaxis are set
        yaxis=True  Boolean, if True the labels for the yaxis are set
    """
    if ax is None:
        ax = plt.gca()

    if xaxis is True:
        xticks = ax.get_xticks() if xticks is None else xticks
        xlabels = [r"$\num[locale={}]{{{:.{}f}}}$".format(locale, tick, x_precision) for tick in xticks]
        ax.set_xticklabels(xlabels)

    if yaxis is True:
        yticks = ax.get_yticks() if yticks is None else yticks
        ylabels = [r"$\num[locale={}]{{{:.{}f}}}$".format(locale, tick, y_precision) for tick in yticks]
        ax.set_yticklabels(ylabels)


def legend(ax=None, linewidth=1.1, loc='best'):
    if ax is None:
        ax = plt.gca()

    leg = ax.legend(loc=loc)
    if linewidth is not None:
        for line in leg.get_lines():
            line.set_linewidth(linewidth)