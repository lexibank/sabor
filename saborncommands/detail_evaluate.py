"""
Evaluate data.
"""

import math
import os
from collections import Counter
from lingpy import *
from tabulate import tabulate
from lexibank_sabor import Dataset as SABOR


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


def report_detail_evaluation(status_counts, file_name):
    metrics = []
    for language, counts in status_counts.items():
        p, r, f, a = prf_(counts['tp'], counts['tn'], counts['fp'], counts['fn'])
        row = [language, counts['tp'], counts['tn'], counts['fp'], counts['fn'], p, r, f, a]
        metrics.append(row)
    report_metrics_table(metrics, file_name)


def get_recipient_languages(wl, donors, family):
    # Need to get list of languages in wl that are not
    # from donor language families.
    # These are recipient languages for borrowings from donors.

    # Much more concise and efficient would be from
    # the language relation, but limited to wordlist.
    donor_families = {fam for (ID, lang, fam)
                      in wl.iter_rows('doculect', family)
                      if lang in donors}
    return sorted({lang for (ID, lang, fam)
                   in wl.iter_rows('doculect', family)
                   if fam not in donor_families})


def evaluate_detection(wl,
                       donors,
                       family="language_family",
                       detection_status="det_status",
                       filename=''):

    languages = get_recipient_languages(wl, donors, family)

    add_detection_status(wl, languages=languages, donors=donors)

    status_counts = {lang: Counter() for lang in languages}
    for idx in wl:
        if wl[idx, 'doculect'] not in languages: continue
        status_counts[wl[idx, 'doculect']][wl[idx, detection_status]] += 1

    overall_counts = Counter()
    for language, counts in status_counts.items():
        overall_counts.update(counts)
    status_counts['Overall'] = overall_counts

    report_detail_evaluation(status_counts, filename)

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


def register(parser):
    parser.add_argument("--file", action="store",
                        default="store/pw-spa-NED-0.10.tsv")
    parser.add_argument("--prefix", action="store",
                        help="Evaluate all files in store with given prefix")
    parser.add_argument("--donor", nargs="*", type=str,
                        default=["Spanish", "Portuguese"],
                        help='Donor language(s).')


def run(args):
    SAB = SABOR()

    if args.prefix:  # Prefix for set of files in store.
        for file in os.listdir(str(SAB.dir / "store")):
            if file.startswith(args.prefix):
                wl = Wordlist(str(SAB.dir / "store" / file))
                evaluate_detection(wl, donors=args.donor, filename=file)
    else:  # Single file.
        wl = Wordlist(str(SAB.dir / args.file))
        evaluate_detection(wl, donors=args.donor, filename=args.file)
