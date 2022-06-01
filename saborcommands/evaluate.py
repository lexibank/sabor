"""
Evaluate data.
"""

import math
import os
from collections import Counter
from lingpy import *
from tabulate import tabulate

from lexibank_sabor import (our_path)


def add_detection_status(wordlist, languages, donors,
                         gold_donor_lang="donor_language",
                         pred_donor_lang="source_language",
                         detection_status="det_status"):
    # Add detection status for each entry.
    # gold_ is donor language from verified source.
    def calc_detection_status(gold_, pred_):
        if pred_ == gold_:
            if gold_: status = 'tp'
            else: status = 'tn'
        else:
            # Policy for donor focused detection status.
            #
            if gold_:
                # Classified as fp because borrowing
                # did not match gold donor.
                if pred_ in donors: status = 'fp'
                # Classified as fn because borrowing not
                # detected but gold donor in donor list.
                elif gold_ in donors: status = 'fn'
                # Classified as tn because borrowing not
                # detected for gold donor not in donor list.
                else: status = 'tn'
            else: status = 'fp'
        return status

    wordlist.add_entries(detection_status, gold_donor_lang, lambda x: None)
    for idx in wordlist:
        if wordlist[idx, "doculect"] in languages:
            wordlist[idx, detection_status] = calc_detection_status(
                wordlist[idx, gold_donor_lang], wordlist[idx, pred_donor_lang])


def prf_(tp, tn, fp, fn, return_nan=False):
    """
    Compute precision, recall, and f-score for tp, tn, fp, fn.
    """

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = math.nan if return_nan else 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = math.nan if return_nan else 0
    if math.isnan(precision) and math.isnan(recall):
        fs = math.nan
    elif math.isnan(precision) or math.isnan(recall):
        fs = math.nan
    elif not precision and not recall:
        fs = 0.0
    else:
        fs = 2 * (precision * recall) / (precision + recall)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    return precision, recall, fs, accuracy


def report_metrics_table(metrics, file_name):
    print()
    print("Detection results for: {}.".format(file_name))
    print(tabulate(metrics,
          headers=['Language', 'tp', 'tn', 'fp', 'fn',
                   'precision', 'recall', 'F1 score', 'accuracy'],
                   tablefmt="pip", floatfmt=".3f"))
    print()


def construct_detail_evaluation(status_counts):
    metrics = []
    for language, counts in status_counts.items():
        p, r, f, a = prf_(counts['tp'], counts['tn'], counts['fp'], counts['fn'])
        row = [language, counts['tp'], counts['tn'], counts['fp'], counts['fn'], p, r, f, a]
        metrics.append(row)

    return metrics


def evaluate_detection(wl,
                       donors,
                       detection_status="det_status",
                       filename='',
                       report=True):

    languages = list(set(wl.cols)-set(donors))

    add_detection_status(wl, languages=languages, donors=donors)

    status_counts = {lang: Counter() for lang in languages}
    for idx in wl:
        if wl[idx, 'doculect'] not in languages: continue
        status_counts[wl[idx, 'doculect']][wl[idx, detection_status]] += 1

    overall_counts = Counter()
    for language, counts in status_counts.items():
        overall_counts.update(counts)
    status_counts['Overall'] = overall_counts

    metrics = construct_detail_evaluation(status_counts)
    summary = [round(val, 3) for val in metrics[-1][1:]]

    if report:
        report_metrics_table(metrics, filename)
        # Print evaluation summary
        tp, fp = overall_counts['tp'], overall_counts['fp']
        tn, fn = overall_counts['tn'], overall_counts['fn']
        print("Overall detection results for: {}".format(filename))
        table = [
                ["identified", tp, fp, tp+fp],
                ["not identified", fn, tn, tn+fn],
                ["total", tp+fn, fp+tn, tp+fp+tn+fn]
                ]
        print(tabulate(table, headers=["", "borrowed", "not borrowed", "total"]))

        filename = filename.removesuffix('.tsv') + '-evaluate'
        wl.output("tsv", filename=filename, prettify=False, ignore="all")


    return summary  # Return summary for use in other app.


def register(parser):
    parser.add_argument("--file")
    parser.add_argument("--prefix",
                        help="Evaluate all files in store with given prefix")
    parser.add_argument(
        "--donor",
        type=str,
        nargs="*",
        default=["Spanish"],
        help="Donor languages for focused analysis."
    )


def run(args):
    if args.prefix:  # Prefix for set of files in store.
        for file in sorted(os.listdir(our_path("store"))):
            if file.startswith(args.prefix):
                wl = Wordlist(our_path("store", file))
                if "donor_language" in wl.columns:
                    evaluate_detection(wl, donors=args.donor, filename=file)
                else:
                    args.log.info("Known donor not specified for {}.".
                                  format(file))
    elif args.file:  # Single file.
        wl = Wordlist(our_path(args.file))
        if "donor_language" in wl.columns:
            evaluate_detection(wl, donors=args.donor, filename=args.file)
        else:
            args.log.info("Known donor not specified for {}.".
                          format(args.file))
    else:
        args.info("No file specified for evaluation.")
