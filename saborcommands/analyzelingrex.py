"""
    Analyze language cognate and borrowing using LingRex.

    John E. Miller, Apr 2, 2022
"""
from pathlib import Path
import argparse
import lingrex
from lingrex import borrowing

import saborcommands.evaluatelingrex as evaluate
import sabor.arguments as argfns
import sabor.accessdb as adb


def analyze_lingrex(dataset,
                    method='lexstat',
                    model='sca',
                    threshold=0.6,
                    ext_threshold=0.45,
                    runs=2000,
                    cluster_method='infomap',
                    store='store',
                    series='',
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

    lingrex.borrowing.external_cognates(
        wl,
        cognates="autocogid",
        family="family",
        ref="autoborid",
        threshold=ext_threshold)

    filename = "lingrex{d}{series}-analysis".\
        format(d='-' if series else '', series=series)
    file_path = Path(store).joinpath(filename).as_posix()
    wl.output('tsv', filename=file_path, ignore='all', prettify=False)

    return filename


def run(args):
    # Get wordlist of entire database.  Not using foreign path option.
    wl = adb.get_wordlist()
    # Clear donor languages from list of language names if all specified.
    args.language = adb.get_language_all(wl, donors=args.donor) \
        if args.language[0] == 'all' else args.language
    # Keep all selected and donor languages.
    wl = adb.select_languages(wl, languages=args.language, donors=args.donor)

    filename = analyze_lingrex(
                            dataset=wl,
                            runs=args.runs,
                            model=args.model,
                            threshold=args.threshold,
                            ext_threshold=args.ext_threshold,
                            method=args.method,
                            cluster_method=args.cluster_method,
                            store=args.store,
                            series=args.series)

    # Report out the results.
    args.infile = filename
    evaluate.run(args)


def register(parser):
    argfns.register_common(parser)
    argfns.register_analysis(parser)

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

    argfns.register_evaluate(parser)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
