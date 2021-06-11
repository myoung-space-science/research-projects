import abc
import argparse
import csv
import datetime
import functools
import operator
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d


def interpolate_to_radius(
    array: np.ndarray,
    target: float,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Interpolate array to a given radius."""
    indices = interpolation_indices(target, r, axis=1)
    i0 = indices[:, 0]
    i1 = indices[:, 1]
    r0 = parameterize(r, i0)
    r1 = parameterize(r, i1)
    theta0 = parameterize(theta, i0)
    theta1 = parameterize(theta, i1)
    phi0 = parameterize(phi, i0)
    phi1 = parameterize(phi, i1)
    weights = compute_interpolation_weights(
        target,
        r0, theta0, phi0,
        r1, theta1, phi1,
    )
    weights[np.asarray(i0 == i1).nonzero()] = 0.0
    f0 = parameterize(array, i0)
    f1 = parameterize(array, i1)
    return np.array(
        [
            (1.0 - w)*f0[t, ...] + w*f1[t, ...]
            for t, w in enumerate(weights)
        ]
    )


def parameterize(array: np.ndarray, indices: Iterable) -> np.ndarray:
    max_length = array.shape[0]
    idx = indices[:max_length] if len(indices) > max_length else indices
    return np.asarray([array[i, j, ...] for i, j in enumerate(idx)])


def compute_interpolation_weights(
    radius: float,
    r0: np.ndarray, theta0: np.ndarray, phi0: np.ndarray,
    r1: np.ndarray, theta1: np.ndarray, phi1: np.ndarray,
) -> np.ndarray:
    """Compute the weights for interpolating to the given radius."""
    x0, y0, z0 = rtp2xyz(r0, theta0, phi0)
    x1, y1, z1 = rtp2xyz(r1, theta1, phi1)
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    ds_sqr = dx**2 + dy**2 + dz**2
    s0_sqr = x0**2 + y0**2 + z0**2
    b = (x0*dx + y0*dy + z0*dz) / ds_sqr
    c = (radius**2 - s0_sqr) / ds_sqr
    return -b + np.sqrt(b**2 + c)


def rtp2xyz(r, t, p):
    """Convert (r, θ, φ) to (x, y, z)."""
    x = r * np.sin(t) * np.cos(p)
    x = zero_floor(x)
    y = r * np.sin(t) * np.sin(p)
    y = zero_floor(y)
    z = r * np.cos(t)
    z = zero_floor(z)
    return (x, y, z)


def zero_floor(v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Round a small number, or array of small numbers, to zero."""
    if v.shape:
        v[np.asarray(np.abs(v) < sys.float_info.epsilon).nonzero()] = 0.0
    else:
        v = 0.0 if np.abs(v) < sys.float_info.epsilon else v
    return v


def interpolation_indices(
    target: float,
    array: np.ndarray,
    axis: int=0,
) -> np.ndarray:
    """Find the indices bounding the interpolation target."""
    return np.apply_along_axis(find_1d_indices, axis, array, target)


def find_1d_indices(array: np.ndarray, target: float) -> Tuple[int, int]:
    """Find the interpolation indices in a 1-D array."""
    leq = array <= target
    lower = np.where(leq)[0].max() if any(leq) else 0
    geq = array >= target
    upper = np.where(geq)[0].min() if any(geq) else len(array)-1
    return lower, upper


def get_colors(name: str, n: int=32) -> List[Tuple[int, ...]]:
    """Get an array of colors drawn from the named color map."""
    cmap = plt.get_cmap(name)
    return [cmap(i) for i in np.linspace(0, 1, n)]


def datestr2sec(datestr: str) -> float:
    """Convert a date/time string to seconds since the start of the year."""
    dt = datetime.datetime.strptime(datestr, '%Y-%jT%H:%M:%S.%f')
    t0 = datetime.datetime(year=dt.year, month=1, day=1)
    return (dt - t0).total_seconds()


def get_psp_energies(path: str) -> List[Tuple[float, float]]:
    """Get a sequence of lower and upper energy-bin bounds."""
    def unpack(line: str) -> List[float]:
        """Unpack a single pair of energies from a file line."""
        return [float(e) for e in line.strip().lstrip(';').split('-')]
    keep = slice(2, 17)
    with Path(path).open('r', newline='') as fp:
        lines = fp.readlines()
        return [tuple(unpack(line)) for line in lines[keep]]


def harmonic_mean(values: Iterable[float]) -> float:
    """Compute the harmonic mean of a set of values."""
    product = functools.reduce(operator.mul, values)
    return np.power(product, 1 / len(values))


def finalize_plot(path: Union[str, Path]=None, verbose: bool=False):
    """Perform any final plot-related actions, including showing or saving."""
    if path is None:
        plt.show()
    else:
        save_path = Path(path).expanduser().resolve()
        if save_path.is_dir():
            save_path.mkdir(exist_ok=True, parents=True)
            save_path = save_path / 'new_plot.png'
        else:
            save_path.parent.mkdir(exist_ok=True, parents=True)
        if verbose:
            print(f"Saving {save_path} ...")
        plt.savefig(save_path)
    plt.close()


class EPREMData(abc.ABC):
    """Abstract Base Class for EPREM simulation results."""
    def __init__(
        self,
        number: int=None,
        directory: Union[str, Path]=None,
        path: Union[str, Path]=None,
    ) -> None:
        """
        The user may create an instance of this class by passing either a stream
        number and the name of a directory containing EPREM output, or a
        complete path to a stream-observer dataset.

        Keyword Parameters
        ---------------------
        number : int

        The stream number to analyze.

        directory : str or pathlib.Path

        The path to a directory containing EPREM stream-observer datasets or
        STAT files with radially-interpolated flux. May be relative and contain
        wildcards.

        path : str or pathlib.Path

        The path to an EPREM stream-observer dataset. May be relative and
        contain wildcards.
        """
        self._dataset = None
        if number and directory:
            self._number = number
            self._directory = Path(directory).expanduser().resolve()
            self._path = None
        elif path:
            self._number = None
            self._directory = None
            self._path = Path(path).expanduser().resolve()
        else:
            message = (
                "You must provide either a stream number and directory"
                " or the path to a dataset."
            )
            raise TypeError(message) from None

    def time(
        self,
        units: str='days',
        offset: float=0.0,
        zero: bool=False,
    ) -> np.ndarray:
        """The EPREM times, in the given units."""
        time = self._times
        scale = {
            'days': 1.0,
            'hours': 24.0,
            'minutes': 1440.0,
            'seconds': 86400.0,
        }
        if zero:
            time -= time[0]
        time += (offset / scale['seconds'])
        return time * scale[units]

    def flux(self, energy: float, **kwargs) -> np.ndarray:
        """The radially-interpolated EPREM flux values from a STAT package.

        This function will interpolate EPREM flux data, which STAT has already
        interpolated to a fixed radius, to the given energy (in MeV). The shape
        of the resultant array will be (# times, # shells).
        """
        flux = self._get_fluxes(**kwargs)
        interp = interp1d(self.energy, flux, axis=-1)
        return interp(energy)

    @property
    def path(self) -> Path:
        """The full path to simulation data."""
        return self._build_path()

    @abc.abstractmethod
    def _build_path(self) -> Path:
        """Get the full path to simulation data."""
        pass

    @property
    def dataset(self) -> Any:
        """This observer's dataset."""
        if self._dataset is None:
            self._dataset = self._get_dataset()
        return self._dataset

    @abc.abstractmethod
    def _get_dataset(self) -> Any:
        """Get an implementation-specific dataset."""
        pass

    @property
    @abc.abstractmethod
    def energy(self) -> np.ndarray:
        """The EPREM energy bins."""
        pass

    @property
    @abc.abstractmethod
    def _times(self) -> np.ndarray:
        """This instance's times."""
        pass

    @abc.abstractmethod
    def _get_fluxes(self, **kwargs) -> np.ndarray:
        """Get fluxes from the given dataset."""
        pass


class STAT(EPREMData):
    """The radially-interpolated STAT data."""
    def _build_path(self) -> Path:
        """Build the full path to simulation data."""
        if self._directory and self._number:
            return self._directory / f'flux_sp00_{self._number:06d}.dat'
        return self._path

    @property
    def energy(self) -> np.ndarray:
        """The EPREM energy bins."""
        return np.loadtxt(self.path, skiprows=6, max_rows=20)

    def _get_dataset(self) -> Any:
        """The pre-computed STAT data."""
        return np.loadtxt(self.path, skiprows=29)

    @property
    def _times(self) -> np.ndarray:
        """This observer's times array."""
        return self.dataset[:, 0] / 24.0

    def _get_fluxes(self, **kwargs) -> np.ndarray:
        """Get this observer's fluxes."""
        return self.dataset[:, 1:]


class EPREM(EPREMData):
    """The EPREM dataset with pre-computed flux."""
    def _build_path(self) -> Path:
        """Build the full path to simulation data."""
        if self._directory and self._number:
            return self._directory / f'flux{self._number:06d}.nc'
        return self._path

    @property
    def energy(self) -> np.ndarray:
        """The EPREM energy bins."""
        return self.dataset.variables['egrid'][0, :]

    @property
    def _times(self) -> np.ndarray:
        """This observer's times array."""
        return self.dataset.variables['time'][:]

    def _get_fluxes(self, radius: float=None) -> np.ndarray:
        """Get this observer's fluxes."""
        if radius is not None:
            return self._flux_at(radius)
        return np.array(self.dataset.variables['flux']).squeeze()

    def _get_dataset(self) -> Dataset:
        """This observer's dataset."""
        return Dataset(self.path, 'r')

    def _flux_at(self, radius: float):
        """Interpolate flux to the given radius."""
        flux = self.dataset.variables['flux']
        r = self.dataset.variables['R'][:]
        theta = self.dataset.variables['T'][:]
        phi = self.dataset.variables['P'][:]
        flux_at_r = interpolate_to_radius(flux, radius, r, theta, phi)
        return flux_at_r.squeeze()

    def flux(self, energy: float, radius: float=None) -> np.ndarray:
        """The EPREM flux values.

        This function will interpolate the EPREM flux output array to the given
        energy (in MeV) and, optionally, to the given radius (in au). If the
        user provides a radius, the shape of the resultant array will be (#
        times,); if not, the shape of the resultant array will be (# times, #
        shells).
        """
        return super().flux(energy, radius=radius)


def get_eprem(dataset_type: str, **kwargs) -> EPREMData:
    """Create the appropriate object to manage an EPREM dataset."""
    dataset_types = {
        'full': EPREM,
        'stat': STAT,
    }
    try:
        Dataset = dataset_types[dataset_type]
        return Dataset(**kwargs)
    except KeyError:
        raise TypeError(f"Unknown dataset type: {dataset_type}")


class PSP:
    """The PSP dataset."""
    def __init__(self, path: Union[str, Path]) -> None:
        """
        Positional Parameters
        ---------------------
        path : str or pathlib.Path

        The path to a text file of PSP data. May be relative and contain
        wildcards.
        """
        self.path = Path(path).expanduser().resolve()
        self._dataset = None
        self._utc = None

    @property
    def _file_sizes(self) -> dict:
        """Information on sizes of data portions in the file.
        
        This property is intended for internal class use only.
        """
        with self.path.open('r') as fp:
            line = fp.readline()
        fields = line.split(',')
        splits = [field.split('=') for field in fields]
        return {s[0].strip('; '): int(s[1].strip('\n ')) for s in splits}

    @property
    def _times_and_fluxes(self) -> np.ndarray:
        """This instance's times and fluxes.

        This property is intended for internal class use. Users may directly
        access times via the ``time`` method or flux via the ``flux`` method.
        """
        return np.loadtxt(
            self.path,
            skiprows=self._file_sizes.get('data_start_row')-1,
            converters={0: datestr2sec},
            encoding='utf-8',
        )

    def energy(self, mode: str='bins') -> np.ndarray:
        """The PSP EPI-Hi energy bins or means."""
        def unpack(line: str) -> List[float]:
            """Unpack a single pair of energies from a file line."""
            return [float(e) for e in line.strip().lstrip(';').split('-')]
        start = self._file_sizes.get('energy_start_row', 1) - 1
        stop = start + self._file_sizes.get('n_bins', 1)
        keep = slice(start, stop)
        with Path(self.path).open('r', newline='') as fp:
            lines = fp.readlines()
            energies = [tuple(unpack(line)) for line in lines[keep]]
        if mode == 'means':
            return np.array([harmonic_mean(e) for e in energies])
        if mode == 'lower':
            return np.array([e[0] for e in energies])
        if mode == 'upper':
            return np.array([e[1] for e in energies])
        return energies

    def time(
        self,
        units: str='days',
        offset: float=0.0,
        zero: bool=False,
    ) -> np.ndarray:
        """The time of PSP observations, in the given units."""
        time = self._times_and_fluxes[:, 0]
        scale = {
            'days': 86400.0,
            'hours': 3600.0,
            'minutes': 60.0,
            'seconds': 1.0,
        }
        if zero:
            time -= time[0]
        time += offset
        return time / scale[units]

    @property
    def flux(self) -> np.ndarray:
        """The PSP EPI-Hi flux observations."""
        return self._times_and_fluxes[:, 1:]

    @property
    def utc(self) -> Iterable[str]:
        """The UTC times listed in the data file."""
        if self._utc is None:
            start = self._file_sizes.get('data_start_row', 1) - 1
            with self.path.open('r', newline='') as fp:
                reader = csv.reader(fp, delimiter=' ')
                rows = [row for row in reader]
                self._utc = [row[0] for row in rows[start:]]
        return self._utc


def get_eprem_flux(
    eprem: EPREMData,
    energies: Iterable=None,
    radius: float=None,
    **time_kwargs,
) -> dict:
    """Create a dictionary containing EPREM time and flux data."""
    return {
        'time': eprem.time(**time_kwargs),
        'flux': np.array([
            eprem.flux(energy, radius=radius)
            for energy in energies
        ]).transpose()
    }


def get_psp_flux(psp: PSP, **time_kwargs) -> dict:
    """Create a dictionary containing PSP time and flux data."""
    return {
        'time': psp.time(**time_kwargs),
        'flux': psp.flux,
    }


def get_time_offsets(
    psp_start: str,
    event_utc: str,
    utc_start: str=None,
) -> Tuple[float, ...]:
    event_time = datetime.datetime.strptime(event_utc, "%Y-%jT%H:%M:%S")
    psp_time = datetime.datetime.strptime(psp_start, "%Y-%jT%H:%M:%S.%f")
    event_offset = (event_time - psp_time).total_seconds()
    if utc_start:
        utc_time = datetime.datetime.strptime(utc_start, "%Y-%m-%d %H:%M:%S")
        return event_offset, (utc_time - psp_time).total_seconds()
    return event_offset, 0.0


def split_key_value_pairs(
    pairs: Iterable,
    totype: type=None,
) -> dict:
    """Split ``'key=value'`` strings into ``{key: value}`` pairs."""
    target = {}
    for pair in pairs:
        k, v = pair.split("=")
        if totype is not None:
            v = totype(v)
        target[k] = v
    return target


class StoreKeyValuePair(argparse.Action):
    """Store key-value pairs from the CLI.
    
    This method adapts the following StackOverflow answer: 
    https://stackoverflow.com/a/42355279/4739101
    """

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        value_type=None,
        **kwargs
    ):
        self._nargs = nargs
        self._type = value_type
        super(
            StoreKeyValuePair,
            self,
        ).__init__(option_strings, dest, nargs=nargs, **kwargs,)

    def __call__(self, parser, namespace, values, option_string=None,):
        _type = str if self._type is None else self._type
        values = split_key_value_pairs(values, totype=_type)
        setattr(namespace, self.dest, values,)

