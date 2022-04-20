"""
    Evaluation with precision, recall, F1 Score and accuracy for
    borrowed words detected by method of cognate intruders
    using LingPy's cluster methods to identify similarities across languages.
    Compare detected and known borrowed word results both overall and
    focusing specifically on donor languages as intruders.

    John E. Miller, Apr 14, 2022
"""
import math
from collections import Counter
from tabulate import tabulate


# ================================
# Utility functions for evaluation.
# =================================
def pred_by_gold(pred, gold):
    """
    Simple stats on tn, tp, fn, fp.

    pred is list of 0, 1 predictions.
    gold is list of corresponding 0, 1 truths.
    """
    assert len(pred) == len(gold)

    tp, tn, fp, fn = 0, 0, 0, 0
    for pred_, gold_ in zip(pred, gold):
        if pred_ == gold_:
            if gold_ == 0:
                tn += 1
            elif gold_ == 1:
                tp += 1
        else:
            if gold_ == 0:
                fp += 1
            elif gold_ == 1:
                fn += 1
    if tn + tp + fn + fp != len(pred):
        print(f"Sum of scores {tn + tp + fn + fp} not equal len(pred) {len(pred)}.")
    return tp, tn, fp, fn


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
        # print("*** NAN ***")
    elif math.isnan(precision) or math.isnan(recall):
        # fs = 0.0
        fs = math.nan
        # print("*** Zero NAN ***")
    elif not precision and not recall:
        fs = 0.0
    else:
        fs = 2 * (precision * recall) / (precision + recall)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    return precision, recall, fs, accuracy


def prf(pred, gold, return_nan=False):
    """
    Compute precision, recall, and f-score for pred and gold.
    """
    tp, tn, fp, fn = pred_by_gold(pred, gold)
    p, r, f, a = prf_(tp, tn, fp, fn, return_nan=return_nan)
    return [tp, tn, fp, fn, p, r, f, a]


# =======================================================
# Calculate precision, recall, F1 score, accuracy metrics
# =======================================================
def calculate_overall_metrics(ids_table):
    # Compute overall metrics - this is weighted average since over individuals.
    pred = [0 if row["CROSS_FAMILY_ID"] == 0 else 1 for row in ids_table]
    loan = [1 if row["BORROWED"] else 0 for row in ids_table]
    return prf(pred, loan)


def calculate_metrics(ids_table, by_fam=False):
    overall = calculate_overall_metrics(ids_table)

    # Option to compute by language or by family.
    selector = "FAMILY" if by_fam else "DOCULECT"
    lang_group = sorted(set(row[selector] for row in ids_table))
    metrics = []
    for lang_member in lang_group:
        pred = [0 if row["CROSS_FAMILY_ID"] == 0 else 1 for row in ids_table
                if row[selector] == lang_member]
        loan = [1 if row["BORROWED"] else 0 for row in ids_table
                if row[selector] == lang_member]
        stats = prf(pred, loan)
        metrics.append([lang_member] + stats)
    metrics.append(['Overall'] + overall)

    return metrics


def report_metrics_table(metrics, by_fam=False, threshold=float('NaN')):
    print()
    print(f"Threshold: {threshold:0.3f}.")
    header0 = 'Language ' + ('Family' if by_fam else '')
    print(tabulate(metrics,
          headers=[header0, 'tp', 'tn', 'fp', 'fn',
                   'precision', 'recall', 'F1 score', 'accuracy'],
                   tablefmt="pip", floatfmt=".3f"))
    total = metrics[-1]
    print(f"Total: borrowed {total[1]+total[4]}, "
          f"inherited {total[2]+total[3]}, "
          f"total {total[1] + total[2] + total[3] + total[4]}")
    print()


def report_metrics(ids_table,
                   donors=None,
                   threshold=float('NaN')):
    # Drop donor words from metrics report when donors given.
    if donors:
        ids_table = [row for row in ids_table if row["DOCULECT"] not in donors]
    metrics = calculate_metrics(ids_table, by_fam=False)
    report_metrics_table(metrics, by_fam=False, threshold=threshold)


# =======================================================
# Calculate cross language family shared cognates.
# =======================================================
def get_id_counts(ids_table):
    # Reporting is by language.
    languages = sorted(set(row["DOCULECT"] for row in ids_table))

    id_counts_table = {}
    for language in languages:
        id_set = set(row["CROSS_FAMILY_ID"] for row in ids_table
                     if row["DOCULECT"] == language and row["CROSS_FAMILY_ID"] != 0)
        id_counts = {language: Counter() for language in languages}

        for row in ids_table:
            if row["CROSS_FAMILY_ID"] in id_set:
                id_counts[row["DOCULECT"]][row["CROSS_FAMILY_ID"]] += 1

        id_counts_table[language] = \
            {lang: [len(counter), sum(counter.values())]
                for lang, counter in id_counts.items()}

    return id_counts_table


def print_id_counts(counts_table, threshold=None):
    keys = [key for key in counts_table]
    for selector in range(2):  # 0=concepts, 1=words.
        sca_id_lines = []
        for key, sca_ids in counts_table.items():
            sca_id_lines.append([key] + [sca_ids[key_][selector] for key_ in keys])
        print()
        unit = 'concepts' if selector == 0 else 'words'
        print(f"Number of {unit} for inferred cross-family borrowing ids "
              f"at threshold: {threshold:0.3f}.")
        print(tabulate(sca_id_lines, headers=['Language']+keys))


def report_id_counts(sca_ids_table,
                     threshold=float('NaN')):
    sca_id_counts_table = get_id_counts(sca_ids_table)
    print_id_counts(sca_id_counts_table, threshold=threshold)
