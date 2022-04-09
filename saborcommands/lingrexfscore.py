"""
    Analyze language cognate and borrowing using LingRex and
    report quality metrics [precision, recall, F score, accuracy].

    John E. Miller, Apr 9, 2022
"""

import os
from pathlib import Path
import argparse
import tempfile
import csv
from lingpy import *

import saborcommands.util as util
import saborcommands.analyzelingrex as ana


def calculate_metrics(ids_table, by_fam=False):
    # [5] is borid, [6] is loan, [0] isf family, [1] is language
    pred = [0 if row[2] == 0 else 1 for row in ids_table]
    loan = [1 if row[3] else 0 for row in ids_table]
    overall = util.prf(pred, loan)

    selector = 0 if by_fam else 1  # calculate by language or by family.
    lang_group = sorted(set(row[selector] for row in ids_table))
    metrics = []
    for lang_member in lang_group:
        pred = [0 if row[2] == 0 else 1 for row in ids_table
                if row[selector] == lang_member]
        loan = [1 if row[3] else 0 for row in ids_table
                if row[selector] == lang_member]
        stats = util.prf(pred, loan)
        metrics.append([lang_member] + stats)
    metrics.append(['Overall'] + overall)

    return metrics


def report_metrics(ids_table, donors=None):
    # Drop donors from metrics report when donors given.
    if donors:
        ids_table = [row for row in ids_table if row[1] not in donors]
    metrics = calculate_metrics(ids_table, by_fam=False)
    util.report_metrics_table(metrics, by_fam=False)


def get_ids_table_for(table, donors=None, any_loan=False):
    family_ = [d['FAMILY'] for d in table]
    if 'LANGUAGE' in table[0]:
        language_ = [d['LANGUAGE'] for d in table]
    else:
        language_ = [d['DOCULECT'] for d in table]
    autoborid_ = [d['AUTOBORID'] for d in table]
    loan_ = [d['BORROWED'] for d in table]
    donor_language_ = [d['DONOR_LANGUAGE'] for d in table]

    # Check that borrowing is from donor language.
    if donors and not any_loan:
        loan_ = [False if not any(donor.startswith(dl)
                                  for donor in donors) else ln
                 for dl, ln in zip(donor_language_, loan_)]

    ids_table = list(zip(family_, language_, autoborid_, loan_))
    return ids_table


def add_donor_entries(store, infile, donors):
    filepath = Path(store).joinpath(infile).as_posix()
    wl = Wordlist(filepath)

    etd = wl.get_etymdict(ref='autoborid')
    # external cognates doesn't seem to cross concepts so assume one concept.
    for cogid, values in etd.items():
        idxs = []
        for v in values:
            if v: idxs += v
        # Code to set borids to zero if not including a donor language.
        languages = [wl[idx, 'doculect'] for idx in idxs]
        has_donor = any(donor.startswith(language)
                        for donor in donors
                        for language in languages)
        if not has_donor:
            for idx in idxs: wl[idx, 'autoborid'] = 0

    return wl


def get_table(store='store', infile=None, donors=None, any_loan=False):
    if not infile.endswith('.tsv'): infile += '.tsv'
    if donors and not any_loan:
        # Get file as wordlist and modify entries for AutoBorId
        wl = add_donor_entries(store, infile, donors=donors)
        # Store in updated file path.
        _, filepath = tempfile.mkstemp(suffix=infile, prefix=None,
                                       dir=store, text=True)
        wl.output('tsv', filename=filepath.removesuffix('.tsv'),
                  ignore='all', prettify=False)

    else:  # No donors or not any_loan, so use file directly.
        filepath = Path(store).joinpath(infile).as_posix()

    table = []
    with open(filepath, newline='') as f:
        dr = csv.DictReader(f, delimiter='\t')
        for d in dr:
            # Fixup values to use Boolean, Float, Int.
            d['BORROWED'] = None if d['BORROWED'] == '' \
                else True if d['BORROWED'] == 'True' else False
            d['AUTOBORID'] = int(d['AUTOBORID'])
            table.append(d)

    if donors and not any_loan: os.remove(filepath)
    # Created temporary file and now removed it.

    return table


def calculate_fscore(filename=None, donors=None, any_loan=False):
    print(f"F score function: {filename}, {donors}, {any_loan}")

    table = get_table(infile=filename, donors=donors, any_loan=any_loan)
    ids_table = get_ids_table_for(table, donors=donors, any_loan=any_loan)
    report_metrics(ids_table, donors)


def register(parser):
    parser.add_argument(
        "--infile",
        type=str,
        default="lingrex-analysis"
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
        "--series",
        type=str,
        default=""
    )
    parser.add_argument(
        "--anyloan",
        action="store_true",
        help='Any loan regardless of donor.'
    )


def run(args):
    args.donor = ["Spanish", "Portuguese"]
    args.method = "lexstat"
    args.model = "sca"

    if args.infile and not args.infile == "None":
        calculate_fscore(
            filename=args.infile,
            donors=args.donor,
            any_loan=args.anyloan
        )
        return

    # Get wordlist of entire database.  Not using foreign path option.
    wl = util.get_wordlist()
    # Clear donor languages from list of language names.
    args.language = util.get_language_all(wl, donors=args.donor)
    # Keep all selected and donor languages.
    wl = util.select_languages(wl, languages=args.language, donors=args.donor)

    filename = ana.analyze_lingrex(
                            dataset=wl,
                            runs=args.runs,
                            model=args.model,
                            threshold=args.threshold,
                            ext_threshold=args.ext_threshold,
                            method=args.method,
                            cluster_method=args.cluster_method,
                            series=args.series)

    calculate_fscore(
                filename=filename,
                donors=args.donor,
                any_loan=args.anyloan)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
