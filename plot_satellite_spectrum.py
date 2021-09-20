import argparse
from pathlib import Path
from typing import *

import numpy as np
import matplotlib.pyplot as plt

from satellites import datasets


def main(
    datafile: str,
    mode: str,
    plotfile: str=None,
    start: str=None,
    stop: str=None,
    low: float=None,
    high: float=None,
    xlim: Iterable[float]=None,
    ylim: Iterable[float]=None,
    verbose: bool=False,
) -> None:
    """Read satellite flux data from an ASCII file."""
    datapath = datasets.full_path(datafile)
    if verbose:
        print(f"Reading {datapath}")
    dataset = datasets.FluxDataset(datapath)
    energies, spectrum = get_arrays(
        dataset,
        mode,
        start=start,
        stop=stop,
        low=low,
        high=high,
    )
    plt.plot(energies, spectrum, 'k.')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"Energy [{dataset.energy_unit}]")
    plt.ylabel(f"Flux [{dataset.flux_unit}]")
    title = dataset.metadata['name']
    plt.title(title, wrap=True)
    plt.figure(num=1, figsize=(12, 12))
    plotpath = create_plotpath(datapath, mode, plotfile)
    if verbose:
        print(f"Saving {plotpath}")
    plt.savefig(datasets.full_path(plotpath))


def get_arrays(
    dataset: datasets.FluxDataset,
    mode: str,
    start: str=None,
    stop: str=None,
    low: float=None,
    high: float=None,
) -> Tuple[np.ndarray]:
    """Get appropriate energy and spectrum arrays."""
    methods = {
        'average flux': dataset.average_flux,
        'fluence': dataset.fluence,
    }
    if mode.lower() not in methods:
        raise ValueError(f"Unknown mode '{mode}'")
    method = methods[mode]
    spectrum = np.array(method(start=start, stop=stop))
    energies = dataset.energies.reduce('arithmetic mean')[low:high]
    return energies, spectrum[list(energies)]


def create_plotpath(datapath: Path, mode: str, plotfile: str=None):
    """Create an appropriate full path based on input."""
    plotmode = mode.replace(' ', '_')
    plotname = f"{datapath.stem}-{plotmode}.png"
    if plotfile is None:
        plotfile = datapath.parent / plotname
    plotpath = datasets.full_path(plotfile)
    if plotpath.is_dir():
        return plotpath / plotname
    return plotpath


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        'datafile',
        help="The path to the file to read. May be relative.",
    )
    p.add_argument(
        'mode',
        help="The type of plot to make. Not case sensitive.",
        choices={'average flux', 'fluence'},
    )
    p.add_argument(
        '-o',
        dest='plotfile',
        help=(
            "The destination of the plot file."
            "\nMay be a full path, a directory, or a name."
            "\nIf a directory, this routine will create a name"
            " from the name of the dataset file and the chosen mode."
            "\nIf a name, this routine will save the plot in the"
            " same directory as the dataset file."
            "Paths and directories may be relative."
        ),
    )
    p.add_argument(
        '--start',
        help="The first date and time of the interval.",
    )
    p.add_argument(
        '--stop',
        help="The final date and time of the interval.",
    )
    p.add_argument(
        '--low',
        help="The lower bound of energies to show.",
        type=float,
    )
    p.add_argument(
        '--high',
        help="The upper bound of energies to show.",
        type=float,
    )
    p.add_argument(
        '--xlim',
        help="The x-axis limits",
        nargs=2,
        type=float,
        metavar=('LO', 'HI'),
    )
    p.add_argument(
        '--ylim',
        help="The y-axis limits",
        nargs=2,
        type=float,
        metavar=('LO', 'HI'),
    )
    p.add_argument(
        '-v',
        '--verbose',
        help="Print runtime messages.",
        action='store_true',
    )
    args = p.parse_args()
    main(**vars(args))
