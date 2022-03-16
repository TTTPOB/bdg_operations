#!/usr/bin/env python3
import pandas as pd
import numpy as np
import click
from yaml import safe_load
from pathlib import Path

bdg_column_name = ["seqnames", "start", "end", "score"]


def read_bdg(file: str) -> pd.DataFrame:
    if file.endswith("bdg"):
        df = pd.read_table(file, header=None, names=bdg_column_name, sep="\t")
    elif file.endswith("h5"):
        df = pd.read_hdf(file, "df")
    else:
        raise ValueError("file must be either bdg or h5")
    return df


@click.command()
@click.option(
    "-i1",
    "--input1",
    help="input bedgraph file 1, theoretically small (experiment)",
    required=True,
)
@click.option(
    "-i2",
    "--input2",
    help="input bedgraph file 2, theoretically big (control)",
    required=True,
)
@click.option("-p", "--prefix", help="output prefix", required=True)
@click.option(
    "--span",
    help="span around the cut site, will be added/subtracted from cut site coordinate in both direction",
    type=int,
    required=True,
)
@click.option(
    "-c",
    "--cut-site",
    type=int,
    help="cut site, please do not include seqname in it",
    required=True,
)
@click.option(
    "-r",
    "--ratio",
    type=float,
    help="ratio of difference for a difference to be considered as a real difference in coverage",
    default=0.01,
)
def main(input1, input2, prefix, span, cut_site, ratio):

    click.echo("Reading input files", err=True)

    bdg1 = read_bdg(input1)
    bdg2 = read_bdg(input2)

    span = (cut_site - span, cut_site + span)
    bdg1 = bdg1.query("start >= @span[0] and end <= @span[1]")
    bdg2 = bdg2.query("start >= @span[0] and end <= @span[1]")

    click.echo("filtered input files with specified range", err=True)

    diff = bdg1.score - bdg2.score

    click.echo("calculated difference", err=True)
    # ratio is compared to the smaller of the two
    pos_coord = np.where(diff > bdg2.score * ratio)[0]
    neg_coord = np.where(diff < -bdg1.score * ratio)[0]

    click.echo("binarized")

    # binarized array
    binarized = np.zeros(len(diff), dtype=np.int8)
    binarized[pos_coord] = 1
    binarized[neg_coord] = -1

    # make a bedgraph
    binarized_bdg = pd.DataFrame(
        {
            "seqnames": bdg1.seqnames,
            "start": bdg1.start,
            "end": bdg1.end,
            "score": binarized,
        }
    )

    click.echo("converted to bedgraph", err=True)

    # write to file with specified prefix
    outfilename = f"{prefix}.bdg"
    ## make dir
    outdir = Path(outfilename).parent
    outdir.mkdir(parents=True, exist_ok=True)

    click.echo("done, writing to file", err=True)

    binarized_bdg.to_csv(outfilename, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
