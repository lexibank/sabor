"""
Evaluate data.
"""

import math
import os
from collections import Counter
from lingpy import *
from tabulate import tabulate


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


def report_detail_evaluation(status_counts, file_name):
    metrics = []
    for language, counts in status_counts.items():
        p, r, f, a = prf_(counts['tp'], counts['tn'], counts['fp'], counts['fn'])
        row = [language, counts['tp'], counts['tn'], counts['fp'], counts['fn'], p, r, f, a]
        metrics.append(row)
    report_metrics_table(metrics, file_name)


def evaluate_file(args):
    excludes = {'Spanish', 'Portuguese'}

    wl = Wordlist(args.file)
    languages = sorted(set(wl.cols) - excludes)

    status_counts = {language: Counter() for language in languages}
    overall_counts = Counter()

    for idx in wl:
        if wl[idx, 'doculect'] not in languages: continue
        status_counts[wl[idx, 'doculect']][wl[idx, 'detect_status']] += 1

    for language, counts in status_counts.items():
        overall_counts.update(counts)
    status_counts['Overall'] = overall_counts

    report_detail_evaluation(status_counts, args.file)

    # Print evaluation summary
    tp, fp = overall_counts['tp'], overall_counts['fp']
    tn, fn = overall_counts['tn'], overall_counts['fn']

    print("Overall detection results for: {}".format(args.file))
    table = [
            ["identified", tp, fp, tp+fp],
            ["not identified", fn, tn, tn+fn],
            ["total", tp+fn, fp+tn, tp+fp+tn+fn]
            ]
    print(tabulate(table, headers=["", "borrowed", "not borrowed", "total"]))


def register(parser):
    parser.add_argument("--file", action="store",
                        default="store/pw-spa-NED-0.10.tsv")
    parser.add_argument("--prefix", action="store",
                        help="Evaluate all files in store with given prefix")


def run(args):
    if args.prefix:
        for file in os.listdir("store"):
            if file.startswith(args.prefix):
                args.file = "store/" + file
                evaluate_file(args)
    else:
        evaluate_file(args)
