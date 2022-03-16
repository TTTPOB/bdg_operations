import pandas as pd
from typing import List, Tuple, Union
import os
import gzip

bdg_column_name = ["seqnames", "start", "end", "score"]
PathLike =  Union[str, os.PathLike]
BdgRow = Tuple[str, int, int, float]


def read_bdg(file: str) -> pd.DataFrame:
    if file.endswith("bdg"):
        df = pd.read_table(file, header=None, names=bdg_column_name, sep="\t")
    elif file.endswith("h5"):
        df = pd.read_hdf(file, "df")
    else:
        raise ValueError("file must be either bdg or h5")
    return df

def read_chrom_size(file: str) -> dict:
    with open(file, "r") as f:
        chrom_size = {}
        for line in f:
            if line.startswith("#"):
                continue
            else:
                chrom, size = line.split("\t")
                chrom_size[chrom] = int(size)
    return chrom_size

def get_bdg_row_generator(file: PathLike)->BdgRow:
    """
    :param file: path to bdg file
    :return: generator of (seqnames, start, end, score)
    """
    if file.endswith("bdg"):
        with open(file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                else:
                    seqnames, start, end, score = line.split("\t")
                    yield seqnames, int(start), int(end), float(score)
    elif file.endswith("h5"):
        df = pd.read_hdf(file, "df")
        for row in df.itertuples():
            yield row.seqnames, row.start, row.end, row.score
    else:
        raise ValueError("file must be either bdg or h5")
