import argparse
import datetime
from typing import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

import eprem.tools as tools


def main(
    psp_path: str,
    stream: int,
    time_unit: str='days',
    utc_start: str=None,
    data_dir: str='./',
    dataset_type: str='full',
    plot_path: str=None,
    xlim: tuple=None,
    ylim: tuple=None,
    title: str=None,
    verbose: bool=False,
):
    """Plot PSP and EPREM proton flux."""

    psp = tools.PSP(psp_path)
    eprem = tools.get_eprem(stream, data_dir, dataset_type)
    psp_start = psp.utc[0]
    event_offset, utc_offset = get_offsets(psp_start, utc_start)
    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap('viridis')
    energies = psp.energy('means')
    colors = [cmap(i) for i in np.linspace(0, 1, len(energies))]
    for (i, energy), color in zip(enumerate(energies), colors):
        plt.plot(
            eprem.time(
                time_unit,
                offset=event_offset-utc_offset,
                zero=True,
            ),
            eprem.flux(energy, radius=0.808122),
            label=f'{energy:.1f} MeV',
            color=color,
        )
        plt.plot(
            psp.time(time_unit, offset=-utc_offset, zero=True),
            psp.flux[:, i],
            marker='o',
            linestyle='',
            color=color,
        )
    plt.yscale('log')
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel(f'{time_unit.capitalize()} since {utc_start or psp_start} UTC')
    plt.ylabel(r'protons / cm$^2$ s sr MeV')
    psp_symbol = Line2D([0], [0], linestyle='None', marker='o', color='black')
    psp_label = f"PSP"
    eprem_symbol = Line2D([0], [0], linestyle='solid', color='black')
    eprem_label = f"EPREM stream {stream}"
    leg = plt.legend(
        [psp_symbol, eprem_symbol],
        [psp_label, eprem_label],
        loc='upper left',
    )
    plt.legend(
        bbox_to_anchor=(1.01, 0.5),
        loc='center left',
        borderaxespad=0
    )
    plt.gca().add_artist(leg)
    if title:
        plt.title(title)
    plt.tight_layout()
    tools.finalize_plot(plot_path, verbose=verbose)


def get_offsets(psp_start: str, utc_start: str=None) -> Tuple[float, ...]:
    event_start = '2020-334T12:45:00'
    event_time = datetime.datetime.strptime(event_start, "%Y-%jT%H:%M:%S")
    psp_time = datetime.datetime.strptime(psp_start, "%Y-%jT%H:%M:%S.%f")
    event_offset = (event_time - psp_time).total_seconds()
    if utc_start:
        utc_time = datetime.datetime.strptime(utc_start, "%Y-%m-%d %H:%M:%S")
        return event_offset, (utc_time - psp_time).total_seconds()
    return event_offset, 0.0


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        'psp_path',
        help=(
            "path to PSP EPI-Hi flux file"
            ";\nmay be relative and contain wildcards"
        ),
    )
    p.add_argument(
        'stream',
        help="EPREM stream number to show",
        type=int,
    )
    p.add_argument(
        '--time_unit',
        help="display units for time",
        default='days',
        choices=('days', 'hours', 'minutes', 'seconds'),
    )
    p.add_argument(
        '--utc_start',
        help="UTC start (YYYY-mm-dd HH:MM:SS) of plot",
    )
    p.add_argument(
        '--data_dir',
        help=(
            "directory containing flux data"
            ";\nsee DATASET_TYPE to declare the data format"
            ";\nmay be relative and contain wildcards"
            " (default: current directory)"
        ),
    )
    p.add_argument(
        '--dataset_type',
        help="The data format",
        choices=('full', 'stat'),
        default='full',
    )
    p.add_argument(
        '--plot_path',
        help=(
            "path to which to save the plot"
            ";\nmay be relative and contain wildcards"
            " (default: show plot on screen)"
        ),
    )
    p.add_argument(
        '--xlim',
        help="set the x-axis limits",
        nargs=2,
        type=float,
        metavar=('LO', 'HI'),
    )
    p.add_argument(
        '--ylim',
        help="set the y-axis limits",
        nargs=2,
        type=float,
        metavar=('LO', 'HI'),
    )
    p.add_argument(
        '--title',
        help="set the plot title",
    )
    p.add_argument(
        '-v',
        '--verbose',
        help="print runtime messages (default: False)",
        action='store_true',
    )
    args = p.parse_args()
    main(**vars(args))
