#!/usr/bin/env python3
import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils.bdg_operation import (
    make_genome_window,
    map_score_in_genome_window,
    fillin_bdg_gap,
)
from utils.read import read_bdg
from multiprocessing import Pool


def process_chromosome(df, genome_window, size):
    df = df.sort_values(by=["start"])
    df = fillin_bdg_gap(df, size)
    df = map_score_in_genome_window(genome_window, df)
    return df


def parameter_generator(bdg_split, genome_window_split, genome_size_dict):
    for group, bdg in bdg_split:
        genome_window = genome_window_split.get_group(group)
        size = genome_size_dict[group]
        yield (bdg, genome_window, size)


@click.command()
@click.option("--input", "-i", type=click.Path(exists=True), required=True)
@click.option("--genome-size", "-g", type=click.Path(exists=True), required=True)
@click.option("--window-size", "-w", type=int, default=50)
@click.option("--output", "-o", type=click.Path(exists=False), required=True)
@click.option("--threads", "-t", type=int, default=1)
def main(input, genome_size, window_size, output, threads):
    click.echo("Reading input bedgraph...", err=True)
    bdg = read_bdg(input)
    click.echo("Reading genome window...", err=True)
    genome_size_df = pd.read_csv(
        genome_size, sep="\t", header=None, names=["seqnames", "size"]
    )
    genome_size_dict = genome_size_df.set_index("seqnames").to_dict()["size"]
    click.echo("making genome window...", err=True)
    genome_window_df = make_genome_window(genome_size_df, window_size)
    genome_window_df_split = genome_window_df.groupby("seqnames")

    click.echo("Mapping score in genome window...", err=True)
    with Pool(processes=threads) as pool:
        result_bdg = pool.starmap(
            process_chromosome,
            parameter_generator(
                bdg.groupby("seqnames"), genome_window_df_split, genome_size_dict
            ),
        )
    click.echo("Concatenating results...", err=True)
    result_bdg = pd.concat(result_bdg).sort_values(by=["seqnames", "start"])

    click.echo("Writing output...", err=True)
    result_bdg.to_csv(output, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
