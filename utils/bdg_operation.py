from typing import Iterable
from numba import njit
from typing import Union
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from multiprocessing import Pool
import itertools

Num = Union[int, float]
bdg_column_name = ["seqnames", "start", "end", "score"]


def get_dense_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: this function only accepts bedgraph file with only one chromosome
    and the bedgraph must be sorted by start position, with no overlap
    """
    # values get, now filling the intervals
    score_array_temp_list = []
    for index, row in tqdm(df.iterrows()):
        row = row.values
        start = row[1]
        end = row[2]
        score = row[3]
        score_array_temp_list.append([score] * (end - start))
    score_array = np.hstack(score_array_temp_list)
    new_bdg_start = np.array(list(range(df["start"].values[0], df["end"].values[-1])))
    new_bdg_end = new_bdg_start + 1
    new_bdg = pd.DataFrame(
        {
            "seqnames": df["seqnames"].iloc[0],
            "start": new_bdg_start,
            "end": new_bdg_end,
            "score": score_array,
        }
    )
    return new_bdg


def merge_adjacent_elemnts(array: np.array) -> pd.DataFrame:
    grouped = []
    for key, group in itertools.groupby(array):
        grouped.append(list(group))
    current_loc = 0
    result_list = []
    for group in grouped:
        next_loc = current_loc + len(group)
        location_and_value = (current_loc, next_loc, group[0])
        result_list.append(location_and_value)
        current_loc = next_loc
    result_df = pd.DataFrame(result_list, columns=["start", "end", "score"])
    return result_df


def merge_equal_score_within_chrom(bdg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge equal score rows
    """
    chrom = bdg_df["seqnames"].iloc[0]
    score = bdg_df["score"].values
    start_archive = bdg_df["start"].values[0]
    merged_df = merge_adjacent_elemnts(score)
    merged_df["seqnames"] = chrom
    merged_df = merged_df[bdg_column_name]
    # offset start and end
    merged_df["start"] = merged_df["start"] + start_archive
    merged_df["end"] = merged_df["end"] + start_archive

    return merged_df


def merge_equal_score(bdg_df):
    """
    Merge equal score rows
    """
    splited_df = bdg_df.groupby("seqnames")
    splited_df = [splited_df.get_group(x) for x in splited_df.groups]
    with Pool(len(splited_df)) as p:
        merged_df_list = p.map(merge_equal_score_within_chrom, splited_df)
    merged_df = pd.concat(merged_df_list)
    return merged_df


@njit
def map_score_in_numpy_array(
    bdg_start_array,
    bdg_end_array,
    bdg_score_array,
    window_start_array,
    window_end_array,
):

    window_number = window_start_array.shape[0]
    window_score_array = np.zeros(window_number)

    bdg_record_number = bdg_score_array.shape[0]

    k = 0
    for i in range(window_number):
        w_start = window_start_array[i]
        w_end = window_end_array[i]
        window_score = 0
        for j in range(k, bdg_record_number):
            bdg_start = bdg_start_array[j]
            bdg_end = bdg_end_array[j]
            bdg_score = bdg_score_array[j]

            new_start = max(w_start, bdg_start)
            new_end = min(w_end, bdg_end)
            if new_end < new_start:
                continue
            window_score += bdg_score * (new_end - new_start)

            if bdg_end >= w_end:
                k = j
                break
        bin_length = new_end - w_start
        # here may be latent bug, please rewrite here later
        # the bin length is not well calculated
        window_score_array[i] = window_score / bin_length
    return window_score_array


def map_score_in_genome_window(
    genome_window: pd.DataFrame, bdg: pd.DataFrame
) -> pd.DataFrame:
    """
    :param genome_window: genome window, must be sorted by start position and have only one chromosome
    :param bdg: bedgraph, must be sorted by start position and have only one chromosome
    :return: a dataframe with the same shape as genome_window, with the sum score of each window
    """
    window_start_array = genome_window["start"].values
    window_end_array = genome_window["end"].values

    bdg_start_array = bdg["start"].values
    bdg_end_array = bdg["end"].values
    bdg_score_array = bdg["score"].values

    window_score_array = map_score_in_numpy_array(
        bdg_start_array,
        bdg_end_array,
        bdg_score_array,
        window_start_array,
        window_end_array,
    )

    result_df = genome_window.copy()
    result_df["score"] = window_score_array
    return result_df


def fillin_bdg_gap(
    bdg: pd.DataFrame, seqlength: int, score: Num = 0, only_fill_last: bool = True
) -> pd.DataFrame:
    """
    :param: bdg: a bedgraph, pandas dataframe, must be sorted, and contains only one chromosome
    :param: window_size: the size of the window, must be positive
    :param: score: the score to fill in the gap
    :return: a bedgraph with the same shape as bdg, with the sum score of each window
    """
    if only_fill_last:
        last_end = bdg["end"].iloc[-1]
        if last_end != seqlength:
            bdg_last_row = pd.DataFrame(
                {
                    "seqnames": bdg["seqnames"].iloc[-1],
                    "start": [last_end],
                    "end": seqlength,
                    "score": score,
                }
            )
            bdg = pd.concat([bdg, bdg_last_row])
        return bdg
    bdg_length = bdg.shape[0]
    prev_end = 0
    fullfilled_row_list = []
    seqname = bdg["seqnames"].iloc[0]
    for _, row in tqdm(bdg.iterrows(), total=bdg_length):
        start = row["start"]
        end = row["end"]
        if start != prev_end:
            new_row = [seqname, prev_end, start, score]
            fullfilled_row_list.append(new_row)
        fullfilled_row_list.append(row.values)
        prev_end = end
    if prev_end != seqlength:
        new_row = [seqname, prev_end, seqlength, score]
        fullfilled_row_list.append(new_row)
    result_df = pd.DataFrame(fullfilled_row_list, columns=bdg_column_name).astype(
        {"seqnames": str, "start": np.int64, "end": np.int64, "score": np.float64}
    )
    return result_df


def fillin_bdg_gap_wrapper(
    splited_bdg,
    chromsize: dict,
    score: Num = 0,
    only_fill_last: bool = True,
    threads=24,
):
    """
    :param: bdg: a bedgraph, pandas dataframe, must be sorted, and contains only one chromosome
    :param: window_size: the size of the window, must be positive
    :param: score: the score to fill in the gap
    :return: a bedgraph with the same shape as bdg, with the sum score of each window
    """
    seqnames = list(splited_bdg.groups.keys())
    with Pool(threads) as p:
        bdg_list = p.starmap(
            fillin_bdg_gap,
            [(splited_bdg.get_group(x), chromsize[x], score, only_fill_last) for x in seqnames],
        )
    return dict(zip(seqnames, bdg_list))


def make_genome_window(genome_size: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    :param genome_size:
    :param window_size:
    :return:
    """
    genome_size_split = genome_size.groupby("seqnames")
    genome_window_dict = {}
    for group, df in genome_size_split:
        size = df["size"].values[0]
        start_array = np.arange(0, size, window_size)
        # ensure the last window is not larger than the genome
        if start_array[-1] == size:
            start_array = start_array[:-1]

        end_array = start_array + window_size
        if end_array[-1] > size:
            end_array[-1] = size
        genome_window = pd.DataFrame(
            {"seqnames": group, "start": start_array, "end": end_array}
        )
        genome_window_dict[group] = genome_window
    genome_window = pd.concat(genome_window_dict.values())

    return genome_window
