import argparse
import datetime
from typing import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from eprem import tools


def main(
    stream_paths: List[str],
    dataset_type: str='full',
):
    """Plot PSP and EPREM flux with uncertainty envelopes."""
    for stream_path in stream_paths:
        eprem = tools.get_eprem(dataset_type, path=stream_path)
        print(eprem.path)


if __name__ == "__main__":
    main(['/path/one', '/path/two'], 'full')