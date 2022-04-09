"""
    Analyze language cognate and borrowing using LingRex.

    John E. Miller, Apr 2, 2022
"""
from pathlib import Path
import argparse
import saborcommands.util as util
# from lingpy import evaluate
import lingrex
from lingrex import borrowing
# from clldutils.markup import Table


def analyze_lingrex(dataset,
                    method='lexstat',
                    model='sca',
                    threshold=0.6,
                    ext_threshold=0.45,
                    runs=2000,
                    # mode='overlap',
                    cluster_method='infomap',
                    # idtype='loose',
                    store='store',
                    series='lingrex',
                    label="",
                    # donors=None,
                    # any_loan=False
                    ):
    wl = dataset

    # See paper, section "4 Results" and section "3.2 Methods".
    # Detect partial cognates:
    lingrex.borrowing.internal_cognates(
        wl,
        family="family",
        partial=False,
        runs=runs,
        ref="autocogid",
        method=method,
        threshold=threshold,
        cluster_method=cluster_method,
        model=model)

    # wl.add_entries("family", "language_family", lambda x: x)
    lingrex.borrowing.external_cognates(
        wl,
        cognates="autocogid",
        family="family",
        ref="autoborid",
        threshold=ext_threshold)

    # Output the evaluation:
    # p1, r1, f1 = evaluate.acd.bcubes(wl, "ucogid", "autocogid", pprint=False)
    # p2, r2, f2 = evaluate.acd.bcubes(wl, "borrowed", "autoborid", pprint=False)
    # print('')
    # with Table("method", "precision", "recall", "f-score",
    #             tablefmt="simple", floatfmt=".4f") as tab:
    #     tab.append(["automated borrowing detection", p2, r2, f2])
    # print('')

    # Mattis: I find format strings with `f"` extremely hard to read in code,
    # and suggest to not use them.
    filename = f"lingrex{'-' if series else ''}{series}{'-' if label else ''}{label}"
    file_path = Path(store).joinpath(filename).as_posix()
    wl.output('tsv', filename=file_path, ignore='all', prettify=False)


def register(parser):
    parser.add_argument(
        "--language",
        nargs="*",
        type=str,
        default=['all'],
        help="'all' or list of languages"
    )
    parser.add_argument(
        "--donor",
        nargs="*",
        type=str,
        default=["Spanish", "Portuguese"],
        help='Donor language(s).',
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sca", "lexstat", "edit-dist", "turchin"],
        default="lexstat",
        help='Scoring method (default: "lexstat").',
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sca", "asjp"],
        default="sca",
        help='Sound class model to transform tokens.'
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help='Threshold to use for internal clustering.',
    )
    parser.add_argument(
        "--ext_threshold",
        type=float,
        default=0.45,
        help='Threshold to use for external clustering.',
    )

    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=["upgma", "infomap"],
        default="upgma",
        help='Method to use in clustering.',
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2000,
        help='Number of runs for lexstat scorer.',
    )
    parser.add_argument(
        "--store",
        type=str,
        default="store",
        help='Directory to store analysis wordlist.'
    )
    parser.add_argument(
        "--series",
        type=str,
        default=""
    )
    parser.add_argument(
        "--label",
        type=str,
        default="analysis"
    )
    # Arguments for reporting.
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help='Directory to write output.'
    )


def run(args):

    # the get_wordlist command must be adjusted so that "language_family" now
    # has "family" as the column name (easy to do via namespace!)
    wl = util.get_wordlist()

    # Sub-select languages based on languages and donors arguments.
    args.language = util.get_language_all(wl) if args.language[0] == 'all' else args.language
    wl = util.select_languages(wl, languages=args.language, donors=args.donor)

    analyze_lingrex(dataset=wl,
                    runs=args.runs,
                    model=args.model,
                    threshold=args.threshold,
                    ext_threshold=args.ext_threshold,
                    method=args.method,
                    cluster_method=args.cluster_method,
                    store=args.store,
                    series=args.series,
                    label=args.label)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
