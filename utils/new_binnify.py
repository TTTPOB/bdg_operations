from multiprocessing.sharedctypes import Value
from .read import read_bdg, read_chrom_size, get_bdg_row_generator, BdgRow
from .bdg_operation import make_genome_window
import pandas as pd
import numpy as np
import re
from collections import deque
from typing import Iterable, List, Tuple, Union, Generator, Dict

def new_binnify(bdg_row_generator: Iterable[BdgRow],binsize:int, chromsize: Dict) -> Generator[BdgRow, None, None]:
    """
    :param bdg_row_generator: generator of (seqnames, start, end, score)
    :param binsize: size of bins
    :param chromsize: dict of chrom size
    :return: generator of (seqnames, start, end, score)
    """
    mainchr_regex = re.compile(r"^chr[0-9XY]+$")
    # filter chromsizes get only mainchr
    chromsize = {k: v for k, v in chromsize.items() if mainchr_regex.match(k)}
    # get genome windows
    genomewindows = make_genome_window(chromsize, binsize)
    # sort genomewindows by seqnames and start
    genomewindows.sort_values(by=["seqnames", "start"], inplace=True)
    # double pointer iterate through genomewindows and bdg_row_generator
    for rownumber, window in genomewindows.iterrows():
        window_chromsome = window.seqnames
        window_start = window.start
        window_end = window.end
        window_score = 0
        for entry_chromosome, entry_start, entry_end, entry_score in bdg_row_generator:
            if entry_chromosome not in chromsize.keys():
                raise ValueError("chromosome {} not in user defined genome size files".format(entry_chromosome))
            new_start = max(window_start, entry_start)
            new_end = min(window_end, entry_end)
            entry_mean_score = entry_score/(entry_end - entry_start)
            if new_end < new_start:
                continue
            window_score +=  entry_mean_score* (new_end - new_start)
            if entry_end >= window_end:
                break