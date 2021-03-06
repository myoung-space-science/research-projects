import abc
import datetime
import dateutil.parser
from pathlib import Path
import operator
import re
from typing import *

import numpy as np
import scipy.stats.mstats as ms


class DataIndex:
    """An object that knows its own data value and array index."""
    def __init__(self, value, index: int) -> None:
        self._value = value
        self._index = int(index)

    def __index__(self) -> int:
        """Use this object as a sequence index."""
        return self._index

    def _add(self, other: Union['DataIndex', int]):
        """Implements a + b."""
        if isinstance(other, DataIndex):
            return self._value + other._value
        return NotImplemented

    def __add__(self, other: Union['DataIndex', int]):
        """Add another instance or get an incremented index."""
        if isinstance(other, int):
            return self._index + other
        return self._add(other)

    def __radd__(self, other: 'DataIndex'):
        """Suppress reflected addition of all other types."""
        return self._add(other)

    def __iadd__(self, other):
        """Disable in-place addition."""
        return NotImplemented

    def _sub(self, other: Union['DataIndex', int]):
        """Implements a - b."""
        if isinstance(other, DataIndex):
            return self._value - other._value
        return NotImplemented

    def __sub__(self, other: Union['DataIndex', int]):
        """Subtract another instance or get a decremented index."""
        if isinstance(other, int):
            return self._index - other
        return self._sub(other)

    def __rsub__(self, other: 'DataIndex'):
        """Suppress reflected subtraction of all other types."""
        return self._sub(other)

    def __isub__(self, other):
        """Disable in-place subtraction."""
        return NotImplemented

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self._value)


class Time(DataIndex): ...


class Energy(DataIndex):
    def __init__(self, value, index: int, unit: str) -> None:
        super().__init__(value, index)
        self._unit = unit

    def _op(self, other: Union['Energy', int], op: Callable, name: str=None):
        if isinstance(other, Energy):
            if other._unit == self._unit:
                return self._value + other._value
            errmsg = (
                f"Cannot {name or 'combine'} instances with different units"
                f" '{self._unit}' and '{other._unit}'"
            )
            raise ValueError(errmsg)
        if isinstance(other, int):
            return self._index + other
        return NotImplemented

    def _add(self, other):
        return self._op(other, operator.add, 'add')

    def _sub(self, other):
        return self._op(other, operator.sub, 'subtract')


class IndexArray(abc.ABC):
    """Interface between dataset arrays and index objects."""
    def __init__(
        self,
        values: Iterable[float],
        indices: Iterable[int]=None,
    ) -> None:
        self._values = values
        self._length = len(self._values)
        self._indices = indices or range(self._length)

    def __len__(self) -> int:
        """The length of this object."""
        return self._length

    def __iter__(self):
        """Iterate over indices."""
        return iter(self._indices)

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Convert to a NumPy array."""
        return np.array(self._values, *args, **kwargs)

    @abc.abstractmethod
    def __getitem__(self, index):
        """Access one or more elements via index methods."""
        return self._values[index]

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return ', '.join(str(time) for time in self._values)


class Times(IndexArray):
    """Times from the dataset."""
    def __init__(
        self,
        strings: Iterable[str],
        indices: Iterable[int]=None,
    ) -> None:
        values = [Times.str2datetime(s) for s in strings]
        super().__init__(values, indices=indices)

    def __getitem__(self, index) -> Union[Time, List[Time]]:
        """Access times via index notation."""
        if isinstance(index, str):
            index = self.str2datetime(index)
        if isinstance(index, datetime.datetime):
            idx = self.closest(index)
            time = self._values[idx]
            return Time(time, idx)
        if isinstance(index, slice):
            start = self.closest(index.start)
            stop = self.closest(index.stop)
            if stop is not None:
                stop += 1  
            step = self.closest(index.step)
            converted = slice(start, stop, step)
            return Times(
                self._values[converted],
                indices=self._indices[converted],
            )
        if isinstance(index, int) and index < 0:
            index += self._length
        time = self._values[index]
        return Time(time, index)

    def closest(self, target: Union[str, datetime.datetime]) -> Optional[int]:
        """Find the available time closest to `target`."""
        if isinstance(target, int):
            return target
        if isinstance(target, datetime.datetime):
            return np.argmin(np.abs(np.array(self._values) - target))
        if isinstance(target, str):
            converted = Times.str2datetime(target)
            return np.argmin(np.abs(np.array(self._values) - converted))

    @staticmethod
    def str2datetime(datestr: str) -> datetime.datetime:
        """Convert a string from ISO format to a datetime."""
        return dateutil.parser.isoparse(datestr)

    @staticmethod
    def compute_timespan(start: str, stop: str):
        """Compute the time span between `start` and `stop`."""
        return Times.str2datetime(stop) - Times.str2datetime(start)


class Energies(IndexArray):
    """Energies from the dataset."""
    def __init__(
        self,
        values: Iterable[float],
        unit: str,
        indices: Iterable[int]=None,
    ) -> None:
        super().__init__(values, indices=indices)
        self.unit = unit

    def reduce(self, mode: str) -> 'Energies':
        """Create a new 1-D instance via the given mode.
        
        Mode options are:
        - 'lower': choose the lower bound of each bin.
        - 'upper': choose the upper bound of each bin.
        - 'arithmetic mean': compute the arithmetic mean of each bin.
        - 'geometric mean': compute the geometric mean of each bin.
        - 'harmonic mean': compute the harmonic mean of each bin.
        """
        if mode == 'lower':
            values = [value[0] for value in self._values]
            return Energies(values, self.unit)
        if mode == 'upper':
            values = [value[1] for value in self._values]
            return Energies(values, self.unit)
        if mode == 'arithmetic mean':
            values = [np.mean(value) for value in self._values]
            return Energies(values, self.unit)
        if mode == 'geometric mean':
            values = [ms.gmean(value) for value in self._values]
            return Energies(values, self.unit)
        if mode == 'harmonic mean':
            values = [ms.hmean(value) for value in self._values]
            return Energies(values, self.unit)
        raise ValueError(f"Unknown reduction mode: '{mode}'")

    def __getitem__(self, index) -> Union[Energy, List[Energy]]:
        """Access energies via index notation."""
        if isinstance(index, float):
            idx = self.closest(index)
            return self._instance(self._values[idx], idx)
        if isinstance(index, slice):
            start = self.closest(index.start)
            stop = self.closest(index.stop)
            if stop is not None:
                stop += 1
            if index.step:
                strcls = f"{self.__class__.__qualname__}"
                message = f"{strcls} does not support slice steps"
                raise TypeError(message) from None
            converted = slice(start, stop)
            return Energies(
                self._values[converted],
                self.unit,
                indices=self._indices[converted],
            )
        energy = self._values[index]
        return self._instance(energy, index)

    def _instance(self, value: Iterable[float], index: int) -> Energy:
        """Convenience method to create a single instance."""
        if index < 0:
            index += self._length
        return Energy(value, index, self.unit)

    def closest(self, target: float) -> Optional[int]:
        """Find the index of the energy bin containing or closest to `target`.

        The behavior of this method depends on the shape of the internal values:
        - If the energy values are stored in a 1-D array (perhaps because the
          user created this instance via `Energies.reduce`) this method searches
          for the minimum of ``|values - target|``.
        - If the energy values are stored in a 2-D array of bin bounds, this
          method searches for the bin whose low-energy bound is less than or
          equal to the target. The motivation for this criterion is that some
          instruments may have non-overlapping bins, so that strictly searching
          for a bin that contains `target` could yield a null result even if
          `target` is within the bounds of the instrument's full energy range.
        """
        if isinstance(target, int):
            return target
        if not isinstance(target, float):
            return
        values = np.array(self._values)
        if values.ndim == 1:
            return np.argmin(np.abs(values - target))
        bottoms = [min(pair) for pair in self._values]
        if target < np.min(bottoms):
            return 0
        if target > np.max(bottoms):
            return -1
        for index, (lo, hi) in enumerate(zip(bottoms[:-1], bottoms[1:])):
            if lo <= target < hi:
                return index


class DatasetIOError(IOError):
    """An error occurred during dataset I/O."""


class FluxDataset:
    """A class that manages flux data stored in an ASCII file."""
    def __init__(
        self,
        filename: str,
        energy_unit: str='MeV',
        comment: str = '#',
        newline: str = '\n',
        fill: str = '=',
    ) -> None:
        self.filepath = full_path(filename)
        self.energy_unit = energy_unit
        self.flux_unit = None
        self._contents = None
        self._header = None
        self._body = None
        self._newline = newline
        self._comment = comment
        self._fill = fill
        self._ncols = None
        self._nchannels = None
        try:
            self.metadata = self._get_metadata()
        except Exception:
            self._raise_io_error("Error while reading metadata")
        try:
            self.energies = self._get_energies()
        except Exception:
            self._raise_io_error("Error while reading energies")
        try:
            self.times, self.fluxes, self.uncertainties = self._parse_data()
        except Exception:
            self._raise_io_error("Error while reading times and fluxes")

    def _raise_io_error(self, basemsg: str=None):
        """Raise an I/O-related error.

        This method will raise `DatasetIOError` with the full file path,
        preceeded by additional text, if provided.
        """
        body = "" if not basemsg else f"{basemsg} for "
        message = f"{body}{self.filepath}"
        raise DatasetIOError(message) from None

    def strip_extra(self, line: str) -> str:
        """Strip comment characters, filler text, etc."""
        tokens = ''.join([self._newline, self._comment, self._fill, ' '])
        return line.strip(tokens)

    def _get_metadata(self) -> Dict[str, str]:
        """Read and store file metadata."""
        datalines = [
            line for line in self.header if 'channel' in line.lower()
        ]
        self._ncols = 2 + len(datalines) # date col + time col + data cols
        fluxlines = [
            line for line in datalines if 'uncertainty' not in line.lower()
        ]
        self._nchannels = len(fluxlines)
        channels = [self._read_channel(fluxline) for fluxline in fluxlines]
        reference = channels[0]['flux unit']
        if any(channel['flux unit'] != reference for channel in channels):
            raise ValueError("Inconsistent flux units.")
        self.flux_unit = reference
        return {
            'name': self.header[2],
            'datefmt': self.header[3].split()[-1],
            'timefmt': self.header[4].split()[-1],
            'channels': channels,
        }

    def _read_channel(self, line: str):
        """Read the metadata for a single energy channel."""
        parts = line.split(':')
        name = parts[1]
        sub = parts[2]
        match = re.search(r"\[([\w\* \(\)\/]+)\]", sub)
        if match:
            egrp = sub[:match.start()].strip().split()
            energy_unit = egrp[1]
            energy_bin = egrp[0]
            flux_unit = sub[match.start():match.end()].strip(' []')
        else:
            energy_bin = energy_unit = flux_unit = None
        return {
            'name': name,
            'energy bin': energy_bin,
            'energy unit': energy_unit,
            'flux unit': flux_unit,
        }

    def _get_energies(self):
        """Read and store energy-bin boundaries."""
        scales = {
            'eV':  {'eV': 1e+0, 'keV': 1e+3, 'MeV': 1e+6},
            'keV': {'eV': 1e-3, 'keV': 1e+0, 'MeV': 1e+3},
            'MeV': {'eV': 1e-6, 'keV': 1e-3, 'MeV': 1e+0},
        }
        scale = scales[self.energy_unit]
        energies = []
        for channel in self.metadata['channels']:
            ebin = channel['energy bin']
            dash = re.search(r"[^.\d]", ebin).group()
            f = scale[channel['energy unit']]
            energies.append([f * float(energy) for energy in ebin.split(dash)])
        return Energies(energies, self.energy_unit)

    def _parse_data(self) -> Tuple[Times, np.ndarray, np.ndarray]:
        """Parse the file data into datetimes and fluxes."""
        datetimes = []
        fluxes = []
        uncertainties = []
        i0 = 2
        i1 = i0 + self.nchannels
        for line in self.body:
            cols = line.split()
            datetimes.append(f"{cols[0]} {cols[1]}")
            flux = [float(v) for v in cols[i0:i1]]
            fluxes.append(flux)
            uncertainties.append(
                [float(u) * v for u, v in zip(cols[i1:], flux)]
            )
        return Times(datetimes), np.array(fluxes), np.array(uncertainties)

    def fluence(self, start: str=None, stop: str=None) -> np.ndarray:
        """Compute the fluence over the given time range.
        
        This method computes fluence by summing fluxes over the indicated
        timespan (by default, the full dataset) and multiplying by the
        observation cadence. This is equivalent to a left Riemann sum with
        constant step size.

        Parameters
        ----------
        start : string
            The date and time of the first record to include in the sum. The
            default value is the first record in the dataset.
        stop : string
            The date and time of the final record to include in the sum. The
            default value is the final record in the dataset.

        Returns
        -------
        ndarray
            A 1-D array of fluences computed over the indicated timespan, as a
            function of energy.
        """
        t0 = self.times[start] if start else None
        t1 = self.times[stop]+1 if stop else None
        cadence = (self.times[1] - self.times[0]).seconds
        fluence = np.nansum(self.fluxes[t0:t1, :], axis=0) * cadence
        uncertainty = np.array([
            np.sqrt(
                np.nansum([df_i ** 2 for df_i in df])
            ) for df in self.uncertainties[t0:t1, :].transpose()
        ])
        return fluence, uncertainty

    def average_flux(self, start: str=None, stop: str=None) -> np.ndarray:
        """Compute the average flux over the given time range.
        
        This method computes average flux by first computing the fluence over
        the appropriate timespan, then dividing my the timespan in seconds. See
        `FluxDataset.fluence` for information about parameters.

        Returns
        -------
        ndarray
            A 1-D array of fluxes averaged over the indicated timespan, as a
            function of energy.
        """
        timespan = (
            self.times[stop or -1] - self.times[start or 0]
        ).total_seconds()
        fluence, uncertainty = self.fluence(start, stop)
        return fluence / timespan, uncertainty / timespan

    @property
    def header(self):
        """The file header."""
        if self._header is None:
            self._header = self._get_header()
        return self._header

    def _get_header(self) -> List[str]:
        """Extract the header from the full file contents."""
        return [
            self.strip_extra(line) for line in self.contents
            if line.startswith('#')
        ]

    @property
    def body(self):
        """The body of the file."""
        if self._body is None:
            self._body = self.contents[len(self.header):]
        return self._body

    @property
    def contents(self):
        """The full file contents. Only reasonable for modest files sizes."""
        if self._contents is None:
            self._contents = self._get_contents()
        return self._contents

    def _get_contents(self) -> List[str]:
        """Read the full file contents."""
        with self.filepath.open('r') as fp:
            return fp.readlines()

    @property
    def ncols(self) -> int:
        """The number of columns in the file body."""
        if self._ncols is None:
            self._ncols = 0
        return self._ncols

    @property
    def nchannels(self) -> int:
        """The number of flux channels in this dataset."""
        if self._nchannels is None:
            self._nchannels = 0
        return self._nchannels


def full_path(filename: Union[str, Path]) -> Path:
    """Expand and resolve `filename`."""
    return Path(filename).expanduser().resolve()
