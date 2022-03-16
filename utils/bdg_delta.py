from typing import Tuple
import pandas as pd
import numpy as np

bdg_column_name = ["seqnames", "start", "end", "score"]


def construct_bdg_from_intervals_and_scores(
    seqname: str, interval_breaks: np.array, scores: np.array
):
    start_array = interval_breaks[:-1]
    end_array = interval_breaks[1:]
    return pd.DataFrame(
        {
            bdg_column_name[0]: [seqname]*len(start_array),
            bdg_column_name[1]: start_array,
            bdg_column_name[2]: end_array,
            bdg_column_name[3]: scores,
        }
    )


def align_two_bdgs(bdg1: pd.DataFrame, bdg2: pd.DataFrame) -> Tuple[pd.DataFrame]:
    bdg1_seqname = np.unique(bdg1["seqnames"].values)
    bdg2_seqname = np.unique(bdg2["seqnames"].values)
    if len(bdg1_seqname) != 1 or len(bdg2_seqname) != 1:
        raise ValueError(
            f"bdg1 and/or bdg2 are not of single chromosome,"
            f" bdg1 {bdg1_seqname}, bdg2 {bdg2_seqname}"
        )
    bdg1_seqname = bdg1_seqname[0]
    bdg2_seqname = bdg2_seqname[0]
    if bdg1_seqname != bdg2_seqname:
        raise ValueError(
            f"bdg1 and bdg2 are not of the same chromosome,"
            f" bdg1 {bdg1_seqname}, bdg2 {bdg2_seqname}"
        )

    if (
        bdg1["start"].values[0] != bdg2["start"].values[0]
        or bdg1["end"].values[-1] != bdg2["end"].values[-1]
    ):
        raise ValueError("The two bedgraphs must have the same start and end position")
    # declare bdg1 interval breaks and scores
    bdg1_interval_breaks = np.append(bdg1["start"].values, bdg1["end"].values[-1])
    bdg1_score_array = bdg1["score"].values

    # same for bdg2
    bdg2_interval_breaks = np.append(bdg2["start"].values, bdg2["end"].values[-1])
    bdg2_score_array = bdg2["score"].values

    # get a new interval breaks array by concatenating bdg1 and bdg2, unique and sort it
    new_interval_breaks = np.concatenate((bdg1_interval_breaks, bdg2_interval_breaks))
    new_interval_breaks = np.unique(new_interval_breaks)
    new_interval_breaks = np.sort(new_interval_breaks)

    # double pointer, get new interval score of bdg1 and bdg2
    previous_break_pos_1 = 0
    previous_break_pos_2 = 0
    new_bdg1_score_array = np.zeros(len(new_interval_breaks) - 1)
    new_bdg2_score_array = np.zeros(len(new_interval_breaks) - 1)
    for i in range(len(new_interval_breaks)):
        current_new_break = new_interval_breaks[i]
        for break_pos_1 in range(previous_break_pos_1, len(bdg1_interval_breaks)):
            if break_pos_1 == len(bdg1_interval_breaks) - 1:
                break
            current_break_1 = bdg1_interval_breaks[break_pos_1]
            if current_break_1 == current_new_break:
                new_bdg1_score_array[i] = bdg1_score_array[break_pos_1]
                previous_break_pos_1 = break_pos_1

        for break_pos_2 in range(previous_break_pos_2, len(bdg2_interval_breaks)):
            if break_pos_2 == len(bdg2_interval_breaks) - 1:
                break
            current_break_2 = bdg2_interval_breaks[break_pos_2]
            if current_break_2 == current_new_break:
                new_bdg2_score_array[i] = bdg2_score_array[break_pos_2]
                previous_break_pos_2 = break_pos_2

    new_bdg1 = construct_bdg_from_intervals_and_scores(
        bdg1_seqname, new_interval_breaks, new_bdg1_score_array
    )
    new_bdg2 = construct_bdg_from_intervals_and_scores(
        bdg1_seqname, new_interval_breaks, new_bdg2_score_array
    )
    return new_bdg1, new_bdg2


def delta_two_bdgs(bdg1: pd.DataFrame, bdg2: pd.DataFrame) -> pd.DataFrame:
    """
    :param: bdg1: a bedgraph, must be aligned with each other
    :param: bdg2: a bedgraph, must be aligned with each other
    :return: a pandas dataframe contains the delta of bdg1 and bdg2
    """
    delta_bdg = bdg1.copy()
    delta_bdg["score"] = bdg1["score"].values - bdg2["score"].values
    return delta_bdg
