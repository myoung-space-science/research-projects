import argparse
import pathlib
import typing

import numpy
import matplotlib.pyplot as plt


class FitFile(typing.NamedTuple):
    """Data from a file created by fit_flux_spectrum.py"""

    energies: numpy.ndarray
    spectrum: numpy.ndarray
    result: numpy.ndarray
    labels: typing.List[str]
    parameters: typing.Dict[str, typing.Any]
    path: pathlib.Path


def read_file(__path: typing.Union[str, pathlib.Path]):
    """Read a fit file created by fit_flux_spectrum.py."""

    path = pathlib.Path(__path).expanduser().resolve()
    with path.open('r', newline='') as fp:
        lines = fp.readlines()
    header = [line for line in lines if line.startswith('#')]
    headlen = len(header)
    parameters = {}
    for line in header[1:-1]:
        name, value, status = line.lstrip('#').rstrip('\n\r').split('=')
        parameters[name.strip()] = {
            'value': float(value),
            'status': status.strip(),
        }
    columns = lines[headlen].rstrip('\n\r').split(',')
    energies, spectrum, result = numpy.loadtxt(
        path,
        delimiter=',',
        skiprows=headlen+1,
        unpack=True,
    )
    return FitFile(
        energies,
        spectrum,
        result,
        columns,
        parameters,
        path,
    )


def main(inputs: typing.Iterable[str], output: str=None) -> None:
    """Plot spectra from one or more file(s) created by fit_flux_spectrum.py."""
    for i, arg in enumerate(inputs):
        path = arg[0]
        label = arg[1] if len(arg) == 2 else path
        fit = read_file(path)
        color = f'C{i}'
        plt.plot(
            fit.energies,
            fit.spectrum,
            color=color,
            linestyle='dotted',
        )
        plt.plot(
            fit.energies,
            fit.result,
            color=color,
            linestyle='solid',
            label=label,
        )
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel(
        'Flux '
        '['
        'Protons'
        r' cm$^{-2}$'
        r' s$^{-1}$'
        r' sr$^{-1}$'
        r' (MeV/nuc)$^{-1}$'
        ']'
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-1, 1e+2])
    plt.ylim([1e-5, 1e+3])
    if not output:
        plt.show()
    else:
        outpath = pathlib.Path(output).expanduser().resolve()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath)
        print(f"File written to {outpath}")
    plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        '-i',
        '--include',
        dest='inputs',
        help=(
            "A fit to include in the plot. Repeat for multiple files."
            "\nEach argument must specify a path to a text file"
            " written by fit_flux_spectrum.py (may be relative)."
            "\nAny argument may optionally include a corresponding plot label."
            "\nThe default plot label is the given path."
        ),
        nargs='+',
        action='append',
    )
    p.add_argument(
        '-o',
        '--output',
        help=(
            "An optional output path (may be relative)."
            "\nIf omitted, this routine will show the plot on the screen."
        )
    )
    args = p.parse_args()
    main(**vars(args))
