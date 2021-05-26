import argparse
from pathlib import Path
import csv
from typing import Dict, Union

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend import Legend
import numpy as np

from eprem import tools, seed


def main(
    spectrum: Union[Path, dict],
    plotpath: Path=None,
    textpath: Path=None,
    free: list=None,
    fixed: dict=None,
    lower: dict=None,
    initial: dict=None,
    upper: dict=None,
):
    """Compute and save a fit to the given fluence spectrum."""
    paths = setup_paths(
        plot=(plotpath, '.png'),
        text=(textpath, '.txt'),
    )
    fit = compute_fit(
        spectrum,
        free=free,
        fixed=fixed,
        lower=lower,
        initial=initial,
        upper=upper,
    )
    print(f"Saving {paths['plot']} ...")
    create_plot(paths['plot'], fit)
    print(f"Saving {paths['text']} ...")
    create_text(paths['text'], fit)


def setup_paths(**paths) -> Dict[str, Path]:
    """Ensure full, read-only paths that exist."""
    return {name: _split(pair) for name, pair in paths.items()}


def _split(pair) -> Path:
    """Helper function for ``setup_path()``."""
    value, suffix = pair if len(pair) == 2 else (pair[0], None)
    try:
        path = Path(value)
    except TypeError:
        path = Path(__file__)
    if suffix is not None:
        path = path.with_suffix(suffix).expanduser().resolve()
    return path


def compute_fit(
    spectrum: Union[Union[str, Path], dict],
    **context
) -> seed.Fitter:
    """Compute the fit and store relevant data in a `Fitter` object."""
    if isinstance(spectrum, (str, Path)):
        data = load_from_file(spectrum)
    elif isinstance(spectrum, dict):
        data = load_from_dict(spectrum)
    else:
        TypeError(spectrum)
    return seed.Fitter(
        energies=data['energies'],
        fluxdata=data['fluences'],
        sigma=data.get('uncertns'),
        **context
    )


def load_from_file(filepath: Union[str, Path]):
    """Load the spectrum from the given file."""
    data = np.loadtxt(filepath, skiprows=6)
    energies = data[:, 0]
    fluences = data[:, 1]
    uncertns = data[:, 2] if data.shape[1] == 3 else None
    return {
        'energies': energies,
        'fluences': fluences,
        'uncertns': uncertns,
    }


def load_from_dict(user: Dict[str, Union[np.ndarray, float]]):
    """Load the spectrum from the given dict."""
    energies = user['E']
    parameters = {
        name: user.get(name, value)
        for name, value in seed.default_values.items()
    }
    fluences = seed.J(energies, **parameters)
    return {
        'energies': energies,
        'fluences': fluences,
    }


def create_plot(path: Path, fit: seed.Fitter):
    """Plot the fluence spectrum and fit."""
    plt.plot(fit.energies, fit.fluxdata, label="data")
    plt.plot(fit.energies, fit.spectrum, label="fit")
    plt.xlim([1e-1, 1e2])
    plt.ylim([1e0, 1e7])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Fluence [Counts / (cm² sr MeV)]')
    original_legend = plt.legend(loc='upper right')
    plt.gca().add_artist(parameter_legend(fit))
    plt.gca().add_artist(original_legend)
    plt.savefig(path)


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


def create_text(path: Path, fit: seed.Fitter):
    """Write fit results to a text file."""
    with path.open('w', newline='') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(csv_header(fit))
        writer.writerow(['Energy (MeV)', 'Data', 'Fit'])
        for row in zip(fit.energies, fit.fluxdata, fit.spectrum.squeeze()):
            writer.writerow(row)


def csv_header(fit: seed.Fitter, comment: str="#") -> list:
    """Build an appropriate header for the text output."""

    def line(name: str, parameter: dict) -> list:
        """Build an individual header line."""
        return [
            f"{comment} {name} = {parameter['value']} = {parameter['status']}"
        ]

    parameters = fit.get_parameter_labels()
    start = [f"{comment} Parameters:"]
    info = [line(*item) for item in parameters.items()]
    end = [f"{comment} "]
    return [start, *info, end]


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        'spectrum',
        help=(
            "path to the file containing a spectrum to fit"
            ";\nmay be relative and contain wildcards"
            "\n\nnote that the Python interface also accepts a dict that"
            "\nprovides an energy array and parameter values, which may be"
            "\nuseful for testing or debugging"
        )
    )
    p.add_argument(
        '--plotpath',
        help="path to which to save a plot of the fit"
    )
    p.add_argument(
        '--textpath',
        help="path to which to write fit results"
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
    args = p.parse_args()
    main(**vars(args))

