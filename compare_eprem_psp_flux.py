import argparse
import datetime
from typing import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from eprem import tools


def main(
    psp_path: str,
    stream: int,
    event_utc: str,
    psp_radius: float=None,
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
    eprem = tools.get_eprem(dataset_type, number=stream, directory=data_dir)
    plt.figure(figsize=(10, 5))
    plt.axes(xlim=xlim, ylim=ylim, title=title, yscale='log')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    data = get_flux_data(
        psp, eprem, event_utc, psp_radius, time_unit, utc_start
    )
    plot_loop(*data)
    time_start = utc_start or psp.utc[0]
    plt.xlabel(f'{time_unit.capitalize()} since {time_start} UTC')
    plt.ylabel(r'protons / cm$^2$ s sr MeV')
    add_legends(stream)
    plt.tight_layout()
    tools.finalize_plot(plot_path, verbose=verbose)


def add_legends(stream: int) -> None:
    """Add energy-bin and plot-marker legends."""
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


def get_flux_data(
    psp: tools.PSP,
    eprem: tools.EPREMData,
    event_utc: str,
    psp_radius: float=None,
    time_unit: str='days',
    utc_start: str=None,
) -> tuple:
    """Plot PSP and EPREM flux at interpolated radius and energies."""
    energies = psp.energy('means')
    psp_start = psp.utc[0]
    event_offset, utc_offset = tools.get_time_offsets(
        psp_start, event_utc, utc_start
    )
    eprem_offset = event_offset - utc_offset
    psp_offset = -utc_offset
    eprem_data = tools.get_eprem_flux(
        eprem,
        energies=energies,
        radius=psp_radius,
        units=time_unit,
        offset=eprem_offset,
        zero=True,
    )
    psp_data = tools.get_psp_flux(
        psp,
        units=time_unit,
        offset=psp_offset,
        zero=True,
    )
    return energies, eprem_data, psp_data


def plot_loop(
    energies: Iterable,
    eprem_data: dict,
    psp_data: dict,
) -> None:
    """The main logic for plotting EPREM v. PSP flux."""
    colors = tools.get_colors('viridis', n=len(energies))
    labels = [f'{energy:.1f} MeV' for energy in energies]
    for (i, label), color in zip(enumerate(labels), colors):
        plt.plot(
            eprem_data['time'],
            eprem_data['flux'][:, i],
            label=label,
            color=color,
        )
        plt.plot(
            psp_data['time'],
            psp_data['flux'][:, i],
            marker='o',
            linestyle='',
            color=color,
        )


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
        'event_utc',
        help="assumed UTC start time of observed event",
    )
    p.add_argument(
        '--psp_radius',
        help=(
            "the radius of PSP at the time of comparison"
            "\n\nif dataset type is 'full', this routine will interpolate EPREM"
            "\noutput to this radius; if dataset type is 'stat', it will ignore"
            "\nthis parameter"
        ),
        type=float,
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
