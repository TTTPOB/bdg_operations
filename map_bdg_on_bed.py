#!/usr/bin/env python3
import numpy as np
import re
import numba
from numba import jit
import pandas as pd
from typing import Iterable, List, Tuple, Union
from multiprocessing import Pool
from pathlib import Path
import os
import sys
import click
import itertools
from utils.read import read_bdg, read_chrom_size
from utils.bdg_operation import fillin_bdg_gap, fillin_bdg_gap_wrapper, map_score_in_genome_window

Num = Union[int, float]
bdg_column_name = ["seqnames", "start", "end", "score"]
bed_column_name = ["seqnames", "start", "end", "name", "score", "strand"]


def get_column_count_of_bed(bed_path: Union[str, os.PathLike]):
    """
    :param: bed_path: path to bed file
    :return: number of columns of bed file
    """
    with open(bed_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            else:
                return len(line.split("\t"))


def read_bed(bed_path: Union[str, os.PathLike]):
    """
    :param: bed_path: path to bed file
    :return: pandas.DataFrame
    """
    bed_column_count = get_column_count_of_bed(bed_path)
    bdg_column_name = bed_column_name[:bed_column_count]
    bed_df = pd.read_csv(bed_path, sep="\t", header=None, names=bdg_column_name)
    return bed_df


def filter_main_chr(df, regex=r"^chr(\d{1,2}|X|Y)$"):
    """
    :param df: pandas.DataFrame
    :param regex: regex pattern
    :return: pandas.DataFrame
    """
    regex = re.compile(regex)
    df_seqnames = df["seqnames"].values
    is_mainchr = np.array([regex.match(seqname) is not None for seqname in df_seqnames])
    filtered_df = df[is_mainchr]
    return filtered_df


def main_parameter_generator(bed_by_chrom, bdg_by_chrom):
    bed_chroms = bed_by_chrom.groups.keys()
    bdg_chroms = bdg_by_chrom.keys()
    # bed chroms must be in bdg chroms
    if not set(bed_chroms).issubset(bdg_chroms):
        raise ValueError("bed chroms must be in bdg chroms")

    for chrom in bdg_chroms:
        yield bed_by_chrom.get_group(chrom), bdg_by_chrom[chrom]


def main(bdg_path, bed_path, out_bdg_path, chrom_size_path, threads=24):
    chrom_size = read_chrom_size(chrom_size_path)

    bdg = read_bdg(bdg_path)
    bed = read_bed(bed_path)

    # filter to keep main chromosome
    filtered_bdg = filter_main_chr(bdg)
    filtered_bed = filter_main_chr(bed)

    # split them into chromosomes
    bdg_by_chrom = filtered_bdg.groupby("seqnames")
    bed_by_chrom = filtered_bed.groupby("seqnames")

    # chromosome names
    seqnames = bdg_by_chrom.groups.keys()

    # for each chromosome, filling the gaps (only the last one)
    bdg_by_chrom = fillin_bdg_gap_wrapper(bdg_by_chrom, chrom_size, threads=threads)

    with Pool(threads) as p:
        mapped_bdg = p.starmap(
            map_score_in_genome_window,
            main_parameter_generator(bed_by_chrom, bdg_by_chrom),
        )
    result_df = pd.concat(mapped_bdg)
    result_df.to_csv(out_bdg_path, sep="\t", header=None, index=None)


@click.command()
@click.option("-b","--bdg-path", type=click.Path(exists=True), required=True)
@click.option("-e","--bed-path", type=click.Path(exists=True), required=True)
@click.option("-o","--out-bdg-path", type=click.Path(), required=True)
@click.option("-s","--chrom-size-path", type=click.Path(exists=True), required=True)
@click.option("-t","--threads", type=int, default=24)
def main_cli(bdg_path, bed_path, out_bdg_path, chrom_size_path, threads):
    main(bdg_path, bed_path, out_bdg_path, chrom_size_path, threads)


if __name__ == "__main__":
    main_cli()
