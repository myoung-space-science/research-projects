import argparse
import csv
import inspect
from pathlib import Path
from typing import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend import Legend

from satellites import datasets
from eprem import seed, tools


def load_from_file(
    filepath: Union[str, Path],
    headlen: int=0,
) -> Dict[str, np.ndarray]:
    """Load the spectrum from the given file."""
    data = np.loadtxt(filepath, skiprows=headlen)
    energies = data[:, 0]
    spectrum = data[:, 1]
    uncertainties = data[:, 2] if data.shape[1] == 3 else None
    return {
        'energies': energies,
        'spectrum': spectrum,
        'uncertainties': uncertainties,
    }


def load_from_dict(
    user: Dict[str, Union[np.ndarray, float]],
) -> Dict[str, np.ndarray]:
    """Load the spectrum from the given dict."""
    energies = np.array(user['E'], ndmin=1)
    parameters = {
        name: user.get(name, value)
        for name, value in seed.default_values.items()
    }
    spectrum = seed.J(energies, **parameters)
    return {
        'energies': energies,
        'spectrum': np.array(spectrum),
    }


def load_from_dataset(
    dataset: datasets.FluxDataset,
    start: str=None,
    stop: str=None,
    low_energy: float=None,
    high_energy: float=None,
) -> Dict[str, np.ndarray]:
    """Load the spectrum from the given dataset."""
    result = dataset.average_flux(start=start, stop=stop)
    if len(result) == 1:
        spectrum = np.array(result)
    elif len(result) == 2:
        spectrum, uncertainties = [np.array(r) for r in result]
    reduced = dataset.energies.reduce('arithmetic mean')
    energies = reduced[low_energy:high_energy]
    spectrum = spectrum[list(energies)]
    if uncertainties.size > 0:
        uncertainties = uncertainties[list(energies)]
    return {
        'energies': np.array(energies),
        'spectrum': spectrum,
        'uncertainties': uncertainties,
    }


def get_arrays(source, **load_kw) -> Tuple[np.ndarray]:
    """Extract energies, spectrum, and uncertainties from `source`."""
    loader = DataLoader(source)
    data = loader(**load_kw)
    energies=data['energies']
    spectrum=data['spectrum']
    uncertainties=data.get('uncertainties')
    arrays = [np.array(energies), np.array(spectrum)]
    if uncertainties is not None:
        arrays.append(np.array(uncertainties))
    return arrays


class DataLoader:
    """A format-agnostic data-loading object."""

    _loaders = {
        (Path,): load_from_file,
        (dict,): load_from_dict,
        (datasets.FluxDataset,): load_from_dataset,
    }

    def __init__(self, source) -> None:
        self.source = source
        self._loader = None

    @property
    def loader(self) -> Callable[..., np.ndarray]:
        """The appropriate data loader for this dataset."""
        if self._loader is None:
            for types, loader in self._loaders.items():
                if isinstance(self.source, types):
                    self._loader = loader
        return self._loader

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        """Load the data and return a mapping to dataset members."""
        if self.loader:
            valid = self._parse(kwargs)
            return self.loader(self.source, **valid)
        raise TypeError(self.source) from None

    def _parse(self, kwargs: dict):
        """Extract known keyword arguments from user options."""
        known = inspect.signature(self.loader).parameters
        return {
            k: kwargs.get(k) for k, v in known.items()
            if v.default is not inspect.Parameter.empty
        }


def plot_spectrum(energies: np.ndarray, spectrum: np.ndarray, **plot_kw):
    """Plot `spectrum` versus `energies`."""
    plt.plot(energies, spectrum, 'k.')
    if xlim := plot_kw.get('xlim'):
        plt.xlim(xlim)
    if ylim := plot_kw.get('ylim'):
        plt.ylim(ylim)
    plt.xscale('log')
    plt.yscale('log')
    if xlabel := plot_kw.get('xlabel'):
        plt.xlabel(xlabel)
    if ylabel := plot_kw.get('ylabel'):
        plt.ylabel(ylabel)
    if title := plot_kw.get('title'):
        plt.title(title, wrap=True)


def compute_fit(
    energies: np.ndarray,
    spectrum: np.ndarray,
    uncertainties: np.ndarray,
    **fit_kw
) -> seed.Fitter:
    """Compute a fit to the spectrum as a function of energy."""
    if uncertainties.size == 0:
        uncertainties = None
    return seed.Fitter(
        energies,
        spectrum,
        sigma=uncertainties,
        **fit_kw
    )


def write_fit(fit: seed.Fitter, opts: dict=None):
    """Write fit results to a text file."""
    path = create_destpath('textdest', 'txt', opts)
    verbose = opts['verbose'] if opts and 'verbose' in opts else False
    if verbose:
        print(f"Saving {path}")
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


def plot_fit(fit: seed.Fitter) -> None:
    """Compute and plot a fit to the spectrum."""
    plt.plot(fit.energies, fit.fluxdata, label="data")
    plt.plot(fit.energies, fit.spectrum, label="fit")
    original_legend = plt.legend(loc='upper right')
    plt.gca().add_artist(parameter_legend(fit))
    plt.gca().add_artist(original_legend)


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


def finalize_plot(opts: dict=None):
    """Perform final plot updates and save the plot to disk."""
    verbose = opts['verbose'] if opts and 'verbose' in opts else False
    path = create_destpath('plotdest', 'png', opts)
    plt.figure(num=1, figsize=(12, 12))
    if verbose:
        print(f"Saving {path}")
    plt.savefig(tools.full_path(path))


def create_destpath(optkey: str, ext: str, opts: dict=None):
    """Create an appropriate destination path based on input."""
    kws = opts or {}
    datafile = kws.get('datafile')
    datapath = tools.full_path(datafile)
    destname = f"{datapath.stem}-fit.{ext.lstrip('.')}"
    userdest = kws.get(optkey)
    if userdest is None:
        userdest = datapath.parent / destname
    destpath = tools.full_path(userdest)
    if destpath.is_dir():
        return destpath / destname
    return destpath


def get_source(opts: dict):
    """Get an appropriate data source from user options."""
    if not (datafile := opts.get('datafile')):
        raise TypeError("Cannot determine data source.")
    verbose = opts.get('verbose', False)
    datapath = tools.full_path(datafile)
    if verbose:
        print(f"Reading data from {datapath}")
    try:
        dataset = datasets.FluxDataset(datapath)
    except datasets.DatasetIOError:
        return datapath
    else:
        return dataset


def parse_opts(opts: dict, valid: Iterable[str]) -> dict:
    """Extract keyword options from all options, based on valid names."""
    kws = opts or {}
    return {k: kws[k] for k in valid if k in kws}


# Can we get these from function signatures?
load_kw_args = {
    'start': {'count': 1, 'type': str},
    'stop': {'count': 1, 'type': str},
    'low_energy': {'count': 1, 'type': float},
    'high_energy': {'count': 1, 'type': float},
    'headlen': {'count': 1, 'type': int},
}
valid_load_kw = list(load_kw_args.keys())


plot_kw_args = {
    'xlim': {'count': 2, 'type': float},
    'ylim': {'count': 2, 'type': float},
    'xlabel': {'count': 1, 'type': str},
    'ylabel': {'count': 1, 'type': str},
    'title': {'count': 1, 'type': str},
}
valid_plot_kw = list(plot_kw_args.keys())


valid_fit_kw = [
    'free',
    'fixed',
    'lower',
    'initial',
    'upper',
]


def main(theory: dict=None, **opts):
    """Plot a flux spectrum with an optional fit."""
    source = theory or get_source(opts)
    load_kw = parse_opts(opts, valid_load_kw)
    energies, spectrum, uncertainties = get_arrays(source, **load_kw)
    plot_kw = parse_opts(opts, valid_plot_kw)
    plot_spectrum(energies, spectrum, **plot_kw)
    fit_kw = parse_opts(opts, valid_fit_kw)
    if 'free' in fit_kw:
        fit = compute_fit(energies, spectrum, uncertainties, **fit_kw)
        write_fit(fit, opts)
        plot_fit(fit)
    finalize_plot(opts)


def parse_cli(parser: argparse.ArgumentParser):
    """Parse and normalize all arguments passed via the CLI."""
    # Current limitations:
    # - Assumes an option without an argument implies `True` (e.g., --verbose)
    known, unknown = parser.parse_known_args()
    full = vars(known)
    valid = {**plot_kw_args, **load_kw_args}
    p = 0
    for arg in unknown[p:]:
        name = arg.lstrip('--')
        if name in valid:
            t = valid[name].get('type', str)
            n = valid[name].get('count', 0)
            if n > 1:
                full[name] = [t(unknown[p+i]) for i in range(1, n+1)]
            elif n == 1:
                full[name] = t(unknown[p+1])
            elif n == 0:
                full[name] = True
            p += n+1
    return full


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        'datafile',
        help="The path to the file to read. May be relative.",
    )
    p.add_argument(
        '--plotdest',
        help=(
            "The destination of the plot file."
            "\nMay be a full path, a directory, or a name."
            "\nIf a directory, this routine will create a name from DATAFILE."
            "\nIf a name, this routine will save the plot in the"
            " same directory as DATAFILE."
            "Paths and directories may be relative."
        ),
    )
    p.add_argument(
        '--textdest',
        help=(
            "The destination of the text file."
            "\nMay be a full path, a directory, or a name."
            "\nIf a directory, this routine will create a name from DATAFILE."
            "\nIf a name, this routine will save the text in the"
            " same directory as DATAFILE."
            "Paths and directories may be relative."
        ),
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
        '-v',
        '--verbose',
        help="Print runtime messages.",
        action='store_true',
    )
    args = parse_cli(p)
    main(**args)

