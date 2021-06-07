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
    stream_paths: List[str],
    dataset_type: str='full',
    psp_radius: float=None,
    target_energy: float=None,
    time_unit: str='days',
    utc_start: str=None,
    plot_path: str=None,
    xlim: tuple=None,
    ylim: tuple=None,
    verbose: bool=False,
) -> None:
    """Plot PSP and EPREM flux with uncertainty envelopes."""
    # Create PSP data object.
    psp = tools.PSP(psp_path)

    # Compute time offsets.
    psp_start = psp.utc[0]
    event_offset, utc_offset = tools.get_time_offsets(psp_start, utc_start)
    eprem_offset = event_offset - utc_offset
    psp_offset = -utc_offset

    # Get PSP time and flux data.
    psp_data = tools.get_psp_flux(
        psp,
        units=time_unit,
        offset=psp_offset,
        zero=True,
    )

    # Compute actual available energy value from target energy.
    psp_energies = psp.energy('means')
    if target_energy is None:
        target_energy = 0.0
    energy_index = np.argmin(np.abs(psp_energies - target_energy))
    energy = psp_energies[energy_index]

    # Get EPREM time and flux from all requested datasets.
    eprem_fluxes = []
    for stream_path in stream_paths:
        eprem = tools.get_eprem(dataset_type, path=stream_path)
        eprem_data = tools.get_eprem_flux(
            eprem,
            energies=[energy],
            radius=psp_radius,
            units=time_unit,
            offset=eprem_offset,
            zero=True,
        )
        eprem_fluxes.append(eprem_data['flux'].squeeze())
    eprem_fluxes = np.array(eprem_fluxes)

    # Compute lower and upper envelope bounds.
    lower = eprem_fluxes.min(axis=0)
    upper = eprem_fluxes.max(axis=0)

    # Set up graphics.
    plt.figure(figsize=(10, 5))
    title = f"Energy = {energy:g} MeV"
    plt.axes(xlim=xlim, ylim=ylim, title=title, yscale='log')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

    # Plot PSP flux against time.
    plt.plot(
        psp_data['time'],
        psp_data['flux'][:, energy_index],
        # marker='o',
        # linestyle='',
    )

    # Plot envelope of EPREM fluxes against time.
    plt.fill_between(
        eprem_data['time'],
        lower,
        upper,
        linestyle='dotted',
        alpha=0.2,
        linewidth=2,
    )

    # Finalize graphics.
    time_start = utc_start or psp.utc[0]
    plt.xlabel(f'{time_unit.capitalize()} since {time_start} UTC')
    plt.ylabel(r'protons / cm$^2$ s sr MeV')
    plt.tight_layout()
    tools.finalize_plot(plot_path, verbose=verbose)


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
        'stream_paths',
        help="path(s) to EPREM data to show",
        nargs='*',
    )
    p.add_argument(
        '--dataset_type',
        help="the data format",
        choices=('full', 'stat'),
        default='full',
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
        '--energy',
        dest='target_energy',
        help="the target energy (in MeV) at which to show fluxes",
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
        '-v',
        '--verbose',
        help="print runtime messages (default: False)",
        action='store_true',
    )
    args = p.parse_args()
    main(**vars(args))
