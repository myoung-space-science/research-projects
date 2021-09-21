import abc
from typing import (
    Union,
    List,
    Dict,
    Callable,
    Iterable,
    Optional,
)

import numpy as np
from scipy.optimize import curve_fit


class _FitDataset:
    """A class to manage data for a fitting scheme."""

    def __init__(
        self,
        xdata: Iterable,
        ydata: Iterable,
    ):
        self._xdata = xdata
        self._ydata = ydata

    def xdata(self, copy: bool=True) -> np.ndarray:
        """The array of independent data.

        Keyword Parameters
        ------------------
        copy : bool

        If true (default), copy this instance's array. This is similar to
        ``numpy.array(..., copy=True)`` except that it copies the array created
        from user data rather than copying the user data when creating the
        array.
        """
        if copy:
            return np.array(self._xdata).copy()
        return np.array(self._xdata)

    def ydata(self, copy: bool=False) -> np.ndarray:
        """The array of dependent data.

        Keyword Parameters
        ------------------
        copy : bool

        If true (default), copy this instance's array. This is similar to
        ``numpy.array(..., copy=True)`` except that it copies the array created
        from user data rather than copying the user data when creating the
        array.
        """
        if copy:
            return np.array(self._ydata).copy()
        return np.array(self._ydata)


class _FitContext:
    """A class to manage a data-fitting context."""

    def __init__(
        self,
        free: List[str]=None,
        fixed: Dict[str, float]=None,
        initial: Dict[str, float]=None,
        lower: Dict[str, float]=None,
        upper: Dict[str, float]=None,
        mode: str=None,
        abort_on_error: bool=True,
        return_error_message: bool=False,
        silent: bool=False,
        **fit_kw,
    ):
        self._free = free
        self._fixed = fixed
        self._initial = initial
        self._lower = lower
        self._upper = upper
        self._mode = mode
        self.abort_on_error = abort_on_error
        self.return_error_message = return_error_message
        self.silent = silent
        self._fit_kw = fit_kw
        self._p0 = None
        self._bounds = None

    @property
    def free(self) -> list:
        """The free parameters to fit."""
        if self._free is None:
            self._free = []
        return self._free

    @property
    def fixed(self) -> dict:
        """Parameters of the target function to hold fixed."""
        if self._fixed is None:
            self._fixed = {}
        return self._dict_of_floats(self._fixed)

    @property
    def initial(self) -> dict:
        """User-supplied initial guesses for free parameters."""
        if self._initial is None:
            self._initial = {}
        return self._dict_of_floats(self._initial)

    @property
    def lower(self) -> dict:
        """User-supplied lower bounds for free parameters."""
        if self._lower is None:
            self._lower = {}
        return self._dict_of_floats(self._lower)

    @property
    def upper(self) -> dict:
        """User-supplied upper bounds for free parameters."""
        if self._upper is None:
            self._upper = {}
        return self._dict_of_floats(self._upper)

    @property
    def mode(self) -> str:
        """Fitting mode."""
        if self._mode is None:
            self._mode = 'standard'
        return self._mode

    @property
    def p0(self) -> list:
        """Array of initial parameter values to pass to the fitter."""
        if self._p0 is None:
            self._p0 = self._build_from_free(
                self.initial, default=1,
            )
        return self._p0

    @property
    def bounds(self,) -> tuple:
        """Tuple of bounds to pass to the fitter."""
        if self._bounds is None:
            self._bounds = (
                self._build_from_free(self.lower, default=-np.inf),
                self._build_from_free(self.upper, default=+np.inf)
            )
        return self._bounds

    @property
    def fit_kw(self) -> dict:
        return {
            'abort_on_error': self.abort_on_error,
            'return_error_message': self.return_error_message,
            'silent': self.silent,
            **self._fit_kw,
        }

    def _dict_of_floats(self, this: dict) -> dict:
        """Ensure a dictionary with floating-point values."""
        return {k: float(v) for k, v in this.items()}

    def _build_from_free(self, this: dict, default: float=None) -> list:
        """Build a list in the same order as the free parameters."""
        result = []
        for key in self.free:
            if key in this:
                result.append(float(this[key]))
            else:
                result.append(default)
        return result


class _FitRunner(abc.ABC):
    """Base class for classes that manage data fitting."""

    def __init__(
        self,
        function: Callable=None,
        xdata: Union[list, np.ndarray]=None,
        ydata: Union[list, np.ndarray]=None,
        guesses: Iterable=None,
        bounds: Iterable=None,
    ):
        self.function = function
        self._xdata = xdata
        self._ydata = ydata
        self._guesses = guesses
        self._bounds = bounds

    @property
    def xdata(self) -> np.ndarray:
        return np.array(self._xdata)

    @property
    def ydata(self) -> np.ndarray:
        return np.array(self._ydata)

    @property
    def guesses(self) -> Iterable:
        return self._guesses

    @property
    def bounds(self) -> Iterable:
        return self._bounds

    @xdata.setter
    def xdata(self, new: Union[list, np.ndarray]):
        self._xdata = new

    @ydata.setter
    def ydata(self, new: Union[list, np.ndarray]):
        self._ydata = new

    @abc.abstractmethod
    def compute(self) -> Optional[str]:
        pass


class CurveFitRunner(_FitRunner):
    """A class to manage fitting data with ``scipy.curve_fit``."""
    def compute(
        self,
        abort_on_error: bool=True,
        return_error_message: bool=False,
        silent: bool=False,
        **fit_kw,
    ) -> Optional[str]:
        """Fit the given function to the given data."""
        try:
            popt, pcov = curve_fit(
                self.function,
                self.xdata,
                self.ydata,
                p0=self.guesses,
                bounds=self.bounds,
                **fit_kw,
            )
        except RuntimeError as err:
            self.popt = None
            self.pcov = None
            if not silent:
                print(f"caught curve_fit RuntimeError: {err}")
            if abort_on_error: raise err
            if return_error_message: return str(err)
            else: return None
        else:
            self.popt = popt
            self.pcov = pcov
            return None


class FlexiFit:
    """A class that provides flexible data fitting.

    This class is capable of fitting a user-defined function with free
    parameters and parameter values chosen at runtime.
    """
    def __init__(
        self,
        function: Callable,
        xdata: Union[list, np.ndarray],
        ydata: Union[list, np.ndarray],
        free: List[str]=None,
        fixed: Dict[str, float]=None,
        initial: Dict[str, float]=None,
        lower: Dict[str, float]=None,
        upper: Dict[str, float]=None,
        mode: str=None,
        abort_on_error: bool=True,
        return_error_message: bool=False,
        silent: bool=False,
        **fit_kw,
    ):
        self.function = function
        self.context = _FitContext(
            free=free,
            fixed=fixed,
            initial=initial,
            lower=lower,
            upper=upper,
            mode=mode,
            abort_on_error=abort_on_error,
            return_error_message=return_error_message,
            silent=silent,
            **fit_kw,
        )
        self.nparameters = len(free)
        self.dataset = _FitDataset(xdata, ydata)
        self._runner = None
        self._values = None
        self._covariance = None
        self._bad_values = None
        self._bad_covariance = None
        self._missing = None
        self._errmsgs = None

    def interface(self, x, *args) -> Union[int, float]:
        """Provide a callable interface to the user function."""
        return self.function(x, **self._parse(args))

    def _parse(self, args) -> dict:
        """Parse variable arguments for fitting."""
        _tmp = self.context.fixed.copy()
        for f, arg in zip(self.context.free, args):
            _tmp[f] = arg
        return _tmp

    @property
    def runner(self) -> CurveFitRunner:
        if self._runner is None:
            self._runner = CurveFitRunner(
                function=self.interface,
                guesses=self.context.p0,
                bounds=self.context.bounds,
            )
        return self._runner

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            self._fit()
        return self._values

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            self._fit()
        return self._covariance

    @property
    def bad_values(self) -> np.ndarray:
        if self._bad_values is None:
            self._bad_values = np.array(
                [self.missing] * self.nparameters
            )
        return self._bad_values

    @property
    def bad_covariance(self) -> np.ndarray:
        if self._bad_covariance is None:
            self._bad_covariance = np.array(
                [[self.missing] * self.nparameters] * self.nparameters
            )
        return self._bad_covariance

    @property
    def missing(self) -> float:
        if self._missing is None:
            self._missing = -np.inf
        return self._missing

    @property
    def errmsgs(self) -> list:
        """Error messages from the fitter."""
        if self._errmsgs is None:
            self._errmsgs = []
        return self._errmsgs

    def _fit(self):
        """Fit the user-defined data.
        
        The idependent-variable array may be a single array with the same shape as the dependent-variable array or an array with first dimension that is the same shape as the dependent-variable array.
        """
        xdata = self.dataset.xdata(copy=True)
        ydata = self.dataset.ydata(copy=True)
        xshape = xdata.shape
        yshape = ydata.shape
        if yshape[0] != xshape[0]:
            message = "X and Y must have the same zeroth dimension"
            raise TypeError(message)
        if ydata.ndim == 1:
            ydata = np.expand_dims(ydata, axis=0)
        values = []
        covariance = []
        errmsgs = []
        for yslice in ydata:
            self.runner.xdata = xdata
            self.runner.ydata = yslice
            _err = self.runner.compute(**self.context.fit_kw)
            if _err:
                _val = self.bad_values
                _cov = self.bad_covariance
            else:
                _val = self.runner.popt
                _cov = self.runner.pcov
            values.append(_val)
            covariance.append(_cov)
            errmsgs.append(_err)
        self._values = np.squeeze(values)
        self._covariance = np.squeeze(covariance)
        self._errmsgs = errmsgs
