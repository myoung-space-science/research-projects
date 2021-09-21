import argparse
from pathlib import Path
from typing import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend import Legend

from satellites import datasets
from eprem import seed, tools


def main(
    datafile: str,
    mode: str,
    plotfile: str=None,
    start: str=None,
    stop: str=None,
    low: float=None,
    high: float=None,
    free: list=None,
    fixed: dict=None,
    lower: dict=None,
    initial: dict=None,
    upper: dict=None,
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
    fit = seed.Fitter(
        energies,
        spectrum,
        free=free,
        fixed=fixed,
        lower=lower,
        initial=initial,
        upper=upper,
    )
    plt.plot(energies, spectrum, 'k.')
    plt.plot(fit.energies, fit.fluxdata, label="data")
    plt.plot(fit.energies, fit.spectrum, label="fit")
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
    original_legend = plt.legend(loc='upper right')
    plt.gca().add_artist(parameter_legend(fit))
    plt.gca().add_artist(original_legend)
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
    return np.array(energies), spectrum[list(energies)]


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


def parameter_legend(fit: seed.Fitter) -> Legend:
    """Build a legend that displays spectrum parameters (fit or not)."""
    labels = fit.get_parameter_labels()
    handles = [
        mlines.Line2D(
            [], [],
            label=rf"{this['string']} = {this['value']}"
        ) for this in labels.values()
    ]
    return plt.legend(
        handles=handles,
        handlelength=0.0,
        loc='lower left',
    )


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
        '--free',
        help="names of free parameters",
        nargs='*',
        choices=tuple(seed.default_values.keys()),
        metavar=("p0", "p1"),
    )
    p.add_argument(
        '--fixed',
        help="key-value pairs of parameters to hold fixed",
        nargs='*',
        value_type=float,
        action=tools.StoreKeyValuePair,
        metavar=("p0=value0", "p1=value1"),
    )
    p.add_argument(
        '--initial',
        help="key-value pairs of initial guesses",
        nargs='*',
        value_type=float,
        action=tools.StoreKeyValuePair,
        metavar=("p0=guess0", "p1=guess1"),
    )
    p.add_argument(
        '--lower',
        help="key-value pairs of lower bounds",
        nargs='*',
        value_type=float,
        action=tools.StoreKeyValuePair,
        metavar=("p0=lower0", "p1=lower1"),
    )
    p.add_argument(
        '--upper',
        help="key-value pairs of upper bounds",
        nargs='*',
        value_type=float,
        action=tools.StoreKeyValuePair,
        metavar=("p0=upper0", "p1=upper1"),
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
