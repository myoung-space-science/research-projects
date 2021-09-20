import datetime
import dateutil.parser
from pathlib import Path
import re
from typing import *

import numpy as np
import scipy.stats.mstats as ms


class Time:
    """A single time from the dataset. Usable as an index."""
    def __init__(self, value: datetime.datetime, index: int) -> None:
        self._value = value
        self._index = int(index)

    def __index__(self) -> int:
        """Use this object as a sequence index."""
        return self._index

    def __add__(self, other: 'Time') -> datetime.timedelta:
        """Add two instances."""
        if isinstance(other, Time):
            return self._value + other._value
        return NotImplemented

    def __sub__(self, other: 'Time') -> datetime.timedelta:
        """Subtract two instances."""
        if isinstance(other, Time):
            return self._value - other._value
        return NotImplemented

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self._value)


class Times:
    """Times from the dataset."""
    def __init__(self, strings: Iterable[str]) -> None:
        self._strings = strings
        self._values = [Times.str2datetime(s) for s in strings]
        self._length = len(self._values)
        self._indices = range(self._length)

    def __len__(self) -> int:
        """The length of this object."""
        return self._length

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Convert to a NumPy array."""
        return np.array(self._values, *args, **kwargs)

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
            return [Time(self._values[i], i) for i in self._indices[converted]]
        if isinstance(index, int) and index < 0:
            index += self._length
        time = self._values[index]
        return Time(time, index)

    def closest(self, target: Union[str, datetime.datetime]) -> Optional[int]:
        """Find the available time closest to `target`."""
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

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return ', '.join(str(time) for time in self._values)


class Energy:
    """A single energy from the dataset. Usable as an index."""
    def __init__(self, value: float, index: int, unit: str) -> None:
        self._value = value
        self._index = int(index)
        self._unit = unit

    def __index__(self) -> int:
        """Use this object as a sequence index."""
        return self._index

    def __add__(self, other: Union['Energy', float]):
        """Add two instances."""
        if isinstance(other, Energy):
            if other._unit == self._unit:
                return self._value + other._value
            raise ValueError(self._bad_units('add', self._unit, other._unit))
        if isinstance(other, int):
            return self._index + other
        return NotImplemented

    def __sub__(self, other: Union['Energy', float]):
        """Subtract two instances."""
        if isinstance(other, Energy):
            if other._unit == self._unit:
                return self._value - other._value
            raise ValueError(
                self._bad_units('subract', self._unit, other._unit)
            )
        if isinstance(other, int):
            return self._index - other
        return NotImplemented

    def _bad_units(self, op, u1, u2):
        """Create an error """
        return (
            f"Cannot {op} instances with different units"
            f" {u1} and {u2}"
        )

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return f"{self._value} [{self._unit}]"


class Energies:
    """Energies from the dataset."""
    def __init__(self, values: Iterable[float], unit: str) -> None:
        self._values = values
        self.unit = unit
        self._length = len(self._values)
        self._indices = range(self._length)

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

    def __len__(self) -> int:
        """The length of this object."""
        return self._length

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Convert to a NumPy array."""
        return np.array(self._values, *args, **kwargs)

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
            step = self.closest(index.step)
            converted = slice(start, stop, step)
            return [
                self._instance(self._values[i], i)
                for i in self._indices[converted]
            ]
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

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return ', '.join(str(time) for time in self._values)


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
        self.metadata = self._get_metadata()
        self.energies = self._get_energies()
        self.times, self.fluxes = self._parse_data()

    def strip_extra(self, line: str) -> str:
        """Strip comment characters, filler text, etc."""
        tokens = ''.join([self._newline, self._comment, self._fill, ' '])
        return line.strip(tokens)

    def _get_metadata(self) -> Dict[str, str]:
        """Read and store file metadata."""
        datacols = [
            line for line in self.header if 'channel' in line.lower()
        ]
        channels = []
        for datacol in datacols:
            parts = datacol.split(':')
            name = parts[1]
            sub = parts[2]
            match = re.search(r"\[([\w\* \(\)\/]+)\]", sub)
            if match:
                egrp = sub[:match.start()].strip().split()
                energy_unit = egrp[1]
                energy_bin = egrp[0]
                flux_unit = sub[match.start():match.end()].strip()
            else:
                energy_bin = energy_unit = flux_unit = None
            channels.append(
                {
                    'name': name,
                    'energy bin': energy_bin,
                    'energy unit': energy_unit,
                    'flux unit': flux_unit,
                }
            )
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

    def _parse_data(self):
        """Parse the file data into datetimes and fluxes."""
        datetimes = []
        fluxes = []
        for line in self.body:
            cols = line.split()
            datetimes.append(f"{cols[0]} {cols[1]}")
            fluxes.append([float(v) for v in cols[2:]])
        return Times(datetimes), np.array(fluxes)

    def fluence(
        self,
        start: str=None,
        stop: str=None,
        low: float=None,
        high: float=None,
    ) -> np.ndarray:
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
        low : float
            The lowest energy to include in the result. The default value is the
            lowest energy in the dataset.
        high : float
            The highest energy to include in the result. The default value is
            highest energy in the dataset.

        Returns
        -------
        ndarray
            A 1-D array of fluences computed over the indicated timespan, as a
            function of energy.
        """
        t0 = self.times[start] if start else None
        t1 = self.times[stop] if stop else None
        e0 = self.energies[low] if low else None
        e1 = self.energies[high] if high else None
        cadence = (self.times[1] - self.times[0]).seconds
        return np.nansum(self.fluxes[t0:t1, e0:e1], axis=0) * cadence

    def average_flux(
        self,
        start: str=None,
        stop: str=None,
        low: float=None,
        high: float=None,
    ) -> np.ndarray:
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
        timespan = (self.times[stop or -1] - self.times[start or 0]).seconds
        return self.fluence(start, stop, low, high) / timespan

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


def full_path(filename: str) -> Path:
    """Expand and resolve `filename`."""
    return Path(filename).expanduser().resolve()
