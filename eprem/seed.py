"""Parameters and tools related to the EPREM seed spectrum."""

import argparse
from typing import Callable, Union

import numpy as np

from .fits import FlexiFit
from .tools import StoreKeyValuePair

equation = (
    r"$J(E,r) = "
    r"\left(J_0/\xi\right)"
    r"\left(E/E_r\right)^{-\gamma_s}"
    r"e^{-E/E_0}$"
)

default_values = {
    'J0': 20.0,
    'xi': 1.0,
    'Er': 1.0,
    'gamma': 1.5,
    'E0': 2.0,
}

tex_strings = {
    'J0': r'$J_0$',
    'xi': r'$\xi$',
    'Er': r'$E_r$',
    'gamma': r'$\gamma$',
    'E0': r'$E_0$',
}

def J(E, J0, xi, Er, gamma, E0):
    """Differential flux version of the EPREM seed spectrum.

    This function is similar to the function defined in
    EPREM/energeticParticlesBoundary.c/sepSeedFunction(), without the radial
    dependence. It is designed only to fit a flux or fluence spectrum as a
    function of energy; the user may then factor in radial dependence.
    """
    amplitude = J0 / xi
    power_law = np.power(E / Er, -1.0*gamma)
    exponential_term = np.exp(-1.0*E / E0)
    return amplitude * power_law * exponential_term


class Fitter:
    """A class to manage fitting the seed spectrum to data."""

    def __init__(
        self,
        energies: Union[list, np.ndarray]=None,
        fluxdata: Union[list, np.ndarray]=None,
        **context_kw,
    ):
        self._energies = energies
        self._fluxdata = fluxdata
        self._function = J
        self._context_dict = {**context_kw}
        self._results = None
        self._values = None
        self._stdevs = None
        self._spectrum = None

    @property
    def energies(self) -> np.ndarray:
        return self._energies

    @property
    def fluxdata(self) -> np.ndarray:
        return self._fluxdata

    @property
    def function(self) -> Callable:
        if self._function is None:
            self._function = J
        return self._function

    @property
    def context_dict(self) -> dict:
        _tmp = self._context_dict.get('fixed')
        _fixed = {} if _tmp is None else _tmp.copy()
        self._context_dict['fixed'] = {**default_values, **_fixed}
        if 'free' not in self._context_dict:
            self._context_dict['free'] = []
        return self._context_dict

    @property
    def results(self) -> FlexiFit:
        if self._results is None:
            self._results = FlexiFit(
                self.function,
                xdata=self.energies,
                ydata=self.fluxdata,
                **self.context_dict,
            )
        return self._results

    @property
    def values(self) -> list:
        if self._values is None:
            defaults = self.results.context.fixed.copy()
            if self.results.values.ndim > 1:
                self._values = [
                    self._store_values(group, defaults)
                    for group in self.results.values
                ]
            else:
                self._values = [
                    self._store_values(self.results.values, defaults)
                ]
        return self._values

    @property
    def stdevs(self) -> list:
        if self._stdevs is None:
            self._stdevs = []
            _default_values = {
                k: 0.0 for k in self.results.context.fixed.keys()
            }
            for c in self.results.covariance:
                _stdevs = np.sqrt(np.diag(c))
                _fit_values = dict(
                    zip(self.context_dict['free'], _stdevs)
                )
                self._stdevs.append(
                    {**_default_values, **_fit_values}
                )
        return self._values

    def _store_values(
        self,
        values: Union[int, float],
        base: dict,
    ) -> dict:
        """Helper method for updating a dict of fit values."""
        _fit_values = dict(zip(self.context_dict['free'], values))
        return {**base, **_fit_values}

    @property
    def spectrum(self) -> np.ndarray:
        if self._spectrum is None:
            self._spectrum = []
            for p in self.values:
                self._spectrum.append(self.function(self.energies, **p))
        return np.array(self._spectrum).transpose()

    @energies.setter
    def energies(self, new: Union[list, np.ndarray]):
        self._energies = new
        self._reset()

    @fluxdata.setter
    def fluxdata(self, new: Union[list, np.ndarray]):
        self._fluxdata = new
        self._reset()

    @function.setter
    def function(self, new: Callable):
        self._function = new
        self._reset()

    @context_dict.setter
    def context_dict(self, new: dict):
        self._context_dict = new
        self._reset()

    def _reset(self,):
        """Reset fit classes."""
        self._context = None
        self._dataset = None
        self._results = None

    def get_parameter_labels(
        self,
        format_code: Union[str, dict]='g',
        index: int=0,
    ) -> dict:
        """Create a dict of parameter properties for labels."""

        labels = {
            k: {'string': v, 'value': None, 'status': None}
            for k, v in tex_strings.items()
        }

        if isinstance(format_code, str):
            _format = {k: format_code for k in labels}
        elif isinstance(format_code, dict):
            _format = {k: format_code.get(k, 'g') for k in labels}
        else:
            _format = {}

        values = self.values[index]
        statuses = self.get_parameter_statuses()
        for k in labels:
            labels[k]['value'] = f"{values[k]:{_format[k]}}"
            labels[k]['status'] = statuses[k]
        return labels

    def get_parameter_statuses(self) -> dict:
        """Create a dict of parameter statuses."""
        statuses = dict.fromkeys(default_values.keys())
        for k in statuses:
            statuses[k] = (
                'free' if k in self.context_dict['free'] else 'fixed'
            )
        return statuses


parser = argparse.ArgumentParser(
    add_help=False,
    allow_abbrev=False,
)
parser.add_argument(
    '--free',
    help="The names of free parameters to fit",
    nargs='*',
    choices=tuple(default_values.keys()),
    metavar=("p0", "p1"),
)
parser.add_argument(
    '--fixed',
    help="Key-value pairs of parameters to hold fixed when fitting",
    nargs='*',
    value_type=float,
    action=StoreKeyValuePair,
    metavar=("p0=value0", "p1=value1"),
)
parser.add_argument(
    '--initial',
    help="Key-value pairs of initial guesses for fit parameters",
    nargs='*',
    value_type=float,
    action=StoreKeyValuePair,
    metavar=("p0=guess0", "p1=guess1"),
)
parser.add_argument(
    '--lower',
    help="Key-value pairs of lower bounds on fit parameters",
    nargs='*',
    value_type=float,
    action=StoreKeyValuePair,
    metavar=("p0=lower0", "p1=lower1"),
)
parser.add_argument(
    '--upper',
    help="Key-value pairs of upper bounds on fit parameters",
    nargs='*',
    value_type=float,
    action=StoreKeyValuePair,
    metavar=("p0=upper0", "p1=upper1"),
)
