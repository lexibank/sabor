"""
    Analyze languages making pairwise alignments with donor languages
    (by default: Spanish and Portuguese). Based on original pairwise
    module which performed analysis and calculation of shared cognates.

    Module was subsequently enhanced to infer borrowing based on shared
    cognates and evaluate detection performance (F1 score, precision,
    recall, accuracy). Detail diagnostic reporting was later added to enable
    troubleshooting of discrepancies in prediction versus actual borrowing.

    This version partitions pairwise into separate analysis, evaluation,
    and diagnostic reporting functions. Intent is that the intermediate
    analysis output file can be made compatible with evaluation and
    diagnostic functions used for multiple language methods.

    Johann-Mattis List, Aug 22, 2021 - original pairwise module.
    John E. Miller, Oct 6, 2021 - borrowing, evaluation, and subsequent.

    John E. Miller, Apr 17, 2022 - separation into analysis function.
"""

from pathlib import Path
import argparse

# from distutils.util import strtobool
from lingpy import *

from itertools import product
from pylexibank import progressbar

import saborcommands.evaluatepairwise as evaluate
import sabor.arguments as argfns
import sabor.accessdb as adb
import sabor.utility as util
import sabor.pairwise as pairwise


# Construct borrowing-bookkeeping (bor_book) intermediate store.
# Config has positional based scorer weights.
def construct_alignments(wl, model, mode, gop, donors, config=None):
    # scoring parameters for alignment.
    scale_value = config["scale"] \
        if config and config.get("scale") else 0.5
    factor_value = config["factor"] \
        if config and config.get("factor") else 0.3

    bor_book = {concept: {} for concept in wl.rows}

    sum_dist = 0.0
    cnt_dist = 0

    for concept in progressbar(wl.rows, desc="pairwise alignment"):
        # Entries correspond to doculect word forms and related data concept.
        entries = wl.get_dict(row=concept)
        # Collect donor entries for this concept.
        donors_entries = {donor: entries.get(donor, []) for donor in donors}

        for doculect in entries:
            # Skip the doculect if a donor language.
            if any((lambda x, y: x in y)(donor, doculect)
                   for donor in donors): continue

            # Construct a dictionary of entries for this concept and doculect.
            # Below we add list of aligned donor words corresponding to this entry.
            bor_book[concept][doculect] = {idx: [] for idx in entries[doculect]}

            for donor in donors:
                # All combinations of donor and doculect entries for this concept.
                # Often just 1 entry each, but can be multiple entries for concept.
                for donor_entry, entry in \
                        product(donors_entries[donor], entries[doculect]):

                    # Combination of donor and doculect entries.
                    donor_word, word = wl[donor_entry, "tokens"], \
                                       wl[entry, "tokens"]
                    pair = Pairwise(donor_word, word)

                    pair.align(
                        distance=True, model=model, mode=mode, gop=gop,
                        scale=scale_value, factor=factor_value)

                    donor_alm, alm, dist = pair.alignments[0]
                    # [0] indexes alignment of the only pair

                    # Alignments indexed by concept, doculect, and entry.
                    # Value is list of candidate donor words as:
                    # (donor, distance from doculect tokens, donor entry).
                    bor_book[concept][doculect][entry] += \
                        [(donor, dist, donor_entry)]
                    sum_dist += dist
                    cnt_dist += 1

    # print(f"Ave dist = {sum_dist/cnt_dist:0.2f}")
    return bor_book


def make_cognate_wordlist(wl, bor_book, thresholds,
                          within_threshold, store, series):

    annotated_wl = pairwise.make_cognate_wordlist(
        wl, bor_book, thresholds, within_threshold)

    filename = f"pairwise{'-' if series else ''}{series}"
    file_path = Path(store).joinpath(filename).as_posix()
    annotated_wl.output('tsv', filename=file_path,
                        ignore='all', prettify=False)

    # Append parameters to file.
    line = ''.join(["# Cluster: pairwise_donor_{}".
                   format(threshold) for threshold in thresholds])
    with open(file_path+'.tsv', 'a+') as fl:
        fl.write("# Created using LingPy and Pairwise analysis module.\n")
        fl.write(line)

    return filename


def report_cognate_counts(wl, bor_book, thresholds, donors):

    for threshold in thresholds:
        proportions = pairwise.count_donor_cognates(
            wl,
            bor_book=bor_book,
            threshold=threshold,
            donors=donors)
        pairwise.print_donor_proportions(
            proportions=proportions,
            threshold=threshold,
            donors=donors)


def run(args):
    filename = args.foreign
    wl = adb.get_wordlist(filename)

    # Sub-select languages based on languages and donors arguments.
    args.language = adb.get_language_all(wl) \
        if args.language[0] == 'all' else args.language
    wl = adb.select_languages(wl, languages=args.language, donors=args.donor)

    bor_book_concept = construct_alignments(
        wl, model=args.model, mode=args.mode, gop=args.gop, donors=args.donor)
    print(f"Len of borrow-bookkeeping concept: {len(bor_book_concept)}.")
    bor_book_language = util.swap_dict_top_levels(bor_book_concept)
    print(f"Len of borrow-bookkeeping language: {len(bor_book_language)}.")

    if args.counts:
        report_cognate_counts(
            wl,
            bor_book=bor_book_language,
            thresholds=args.threshold,
            donors=args.donor)

    filename = make_cognate_wordlist(
        wl,
        bor_book=bor_book_concept,
        thresholds=args.threshold,
        within_threshold=args.within,
        store=args.store,
        series=args.series)

    # Evaluate borrowing detection.
    args.infile = filename
    evaluate.run(args)


def register(parser):
    parser.add_argument(
        "--model",
        type=str,
        choices=["sca", "asjp"],
        default="sca",
        help='Sound class model to transform tokens.'
    )
    parser.add_argument(
        "--threshold",
        nargs="*",
        type=float,
        default=[0.4],
        help='Threshold(s) to use with pairwise alignment method.',
    )
    parser.add_argument(
        "--within",
        type=float,
        default=0.2,
        help='Within family threshold for pairwise alignment method.',
    )
    # parser.add_argument(
    #     "--limit",
    #     type=float,
    #     default=None,
    #     help="Limit to use for reporting words and donor candidate distances."
    # )
    # parser.add_argument(
    #     "--status",
    #     type=str,
    #     default='ntn',
    #     choices=[e.name.lower() for e in diag.PredStatus],
    #     help="Status mask to use for reporting borrowed word detection status."
    # )
    parser.add_argument(
        "--mode",
        type=str,
        default="overlap",
        choices=["global", "local", "overlap"],
        help='Alignment mode.',
    )
    parser.add_argument(
        "--gop",
        type=float,
        default=-1.0,
        help='Gap open penalty.'
    )
    # parser.add_argument(
    #     "--pbs",
    #     action="store_true",
    #     help='Use positional based scoring.'
    # )
    # parser.add_argument(
    #     "--min_len",
    #     type=int,
    #     default=1,
    #     help='Minimum length of match for position based scoring.'
    # )

    parser.add_argument(
        "--counts",
        action="store_true",
        help='Show donor cognate counts from analysis.'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help='Directory to write output.'
    )
    parser.add_argument(
        "--foreign",
        type=str,
        default=None,
        help="Filename of flat wordlist for analysis from foreign-tables directory."
    )

    argfns.register_common(parser)
    argfns.register_evaluate(parser)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
