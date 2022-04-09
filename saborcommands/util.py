"""
    Shared routines for keypano.

    John E. Miller, Sep 7, 2021
"""
import tempfile
from pathlib import Path
import math
from enum import Enum
import regex as re
import csv
import sys
from lingpy.sequence.sound_classes import token2class
from tabulate import tabulate

from lingpy import *
from lingpy.compare.util import mutual_coverage_check
from lingpy.compare.sanity import average_coverage
from lexibank_sabor import Dataset as sabor


# Positional scorers
##
def adjust_dist_for_length(dist, alm, max_len=0, min_len=0):
    # If local mode reduces the alignment size too much,
    # the distance measure may not be reliable.
    # Set distance to 1.0 if alignment cropped to less than minimum length.
    # print(f"Entering dist {dist}, min_len {min_len}, max_len {max_len}, len_alm {len(alm)}")
    if min_len <= 1 or max_len <= 0: return dist  # No test for cropped lenth < mimimum.
    dist = dist if len(alm) >= min_len or len(alm) == max_len else 1.0
    # print(f"Leaving dist {dist}, min_len {min_len}, max_len {max_len}, len_alm {len(alm)}")
    return dist


def position_based_scoring(
        almA, almB, gop=-1, weights=None, scorer=None, model="sca", max_len=0, min_len=1):

    weights = weights or {"C": 1, "c": 0.8, "V": 0.6, "v": 0.4, "_": 0}
    scorer = scorer or rc("model").scorer

    # get consensus string
    consensus, clsA, clsB = [], [], []
    for a, b in zip(almA, almB):
        if a == "-":
            consensus += [b]
            clsA += ["-"]
            clsB += [token2class(b, model)]
        elif b == "-":
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += ["-"]
        else:
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += [token2class(b, model)]
    # get prosodic censensus
    prostring = prosodic_string(consensus, _output="CcV")
    prostringA = prosodic_string([x for x in almA if x != "-"], _output="CcV")
    prostringB = prosodic_string([x for x in almB if x != "-"], _output="CcV")

    # score
    scores, scoresA, scoresB = [], [], []
    for a, b, p in zip(clsA, clsB, prostring):
        if "-" in [a, b]:
            # Doesn't use scorer for -, rather use gop.
            scores += [gop * weights[p]]
        else:
            scores += [scorer[a, b] * weights[p]]

    for a, p in zip([x for x in clsA if x != "-"], prostringA):
        scoresA += [scorer[a, a] * weights[p]]
    for b, p in zip([x for x in clsB if x != "-"], prostringB):
        scoresB += [scorer[b, b] * weights[p]]

    scores_sum = sum(scoresA) + sum(scoresB)
    if scores_sum == 0: return 1.0
    dist = 1-(2 * sum(scores))/scores_sum
    adj_dist = adjust_dist_for_length(dist, almA, max_len=max_len, min_len=min_len)
    # print(f"Dist={dist}, Sum={sum(scores)}, Score: {scores}, "
    #       f"SCA: {clsA}, {clsB}, A {scoresA}, B {scoresB}")
    return adj_dist


def relative_position_based_scoring(
        almA, almB, gop=-2, weights=None, model="sca", min_len=4, wt_fn=2):
    # Starting with previous positional scorer base.

    weights = weights or {"C": 1, "c": 0.8, "V": 0.6, "v": 0.4, "_": 0}
    scorer = rc("model").scorer
    # print('scorer', scorer)

    # get consensus and token alignment class strings
    consensus, clsA, clsB = [], [], []
    for a, b in zip(almA, almB):
        if a == "-":
            consensus += [b]
            clsA += ["-"]
            clsB += [token2class(b, model)]
        elif b == "-":
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += ["-"]
        else:
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += [token2class(b, model)]
    # print('Classes', clsA, clsB)

    # get prosodic censensus and alignment strings.
    prostring = prosodic_string(consensus, _output="CcV")
    prostringA = prosodic_string([x for x in almA if x != "-"], _output="CcV")
    prostringB = prosodic_string([x for x in almB if x != "-"], _output="CcV")
    # print('* prosodic', prostringA, prostringB)

    # Calculate relative weights.
    # def find_seq_start(str, start):
    left, right, size = get_core_bounds(almA, almB)
    # print(f"left {left}, right {right}, size {size}.")
    if size >= min_len:
        relative_weights = calculate_alm_wts(left, right, len(almA), wt_fn=wt_fn)
    else:
        relative_weights = [1.0]*len(almA)

    # print(f"rel wts: {relative_weights}")

    # calculate score.
    scores, scoresA, scoresB = [], [], []
    for i, (a, b, p) in enumerate(zip(clsA, clsB, prostring)):
        if "-" in [a, b]:
            # Doesn't use weight for -, rather use gop.
            scores += [gop * weights[p] * relative_weights[i]]
        else:
            scores += [scorer[a, b] * weights[p] * relative_weights[i]]

    for i, (a, p) in enumerate(zip([x for x in clsA if x != "-"], prostringA)):
        scoresA += [scorer[a, a] * weights[p] * relative_weights[i]]
    for i, (b, p) in enumerate(zip([x for x in clsB if x != "-"], prostringB)):
        scoresB += [scorer[b, b] * weights[p] * relative_weights[i]]

    scores_sum = sum(scoresA) + sum(scoresB)
    if scores_sum == 0: return 1.0
    dist = 1-(2 * sum(scores))/scores_sum
    # print('scores', scores)
    # print('* scores *', 2*sum(scores), scores_sum)

    return dist


def find_seq_start(alm, dir):
    it = range(0, len(alm), 1) if dir == 1 else range(-1, -len(alm)-1, -1)
    for i in it:  # range(0, len(alm), dir):
        if alm[i] not in ['-', '_', '+']:
            return i
    return None  # Shouldn't happen since words should contain a sequence.


def get_core_bounds(almA, almB):
    left = max(find_seq_start(almA, 1), find_seq_start(almB, 1))
    if not find_seq_start(almA, -1) or not find_seq_start(almB, -1):
        print(f"No start: {almA}, {find_seq_start(almA, -1)}, "
              f"{almB}, {find_seq_start(almB, -1)}")
    right = min(find_seq_start(almA, -1), find_seq_start(almB, -1))
    size = len(almA)+right-left+1
    return left, right, size


def calculate_alm_wts(left, right, ln, wt_fn=2):
    # Calculate wt vector for score.
    mul = 1.0/(2.0 ** (1./wt_fn))
    wts = [1] * ln
    wt = 1.0
    for i in range(left-1, -1, -1):
        wt *= mul
        wts[i] = wt
    wt = 1.0
    for i in range(right+1, 0, +1):
        wt *= mul
        wts[i] = wt
    return wts


##################################################
# Data functions
##
def get_language_all(wordlist, donors=None):
    donors = donors if donors else ["Spanish", "Portuguese"]
    languages = wordlist.cols
    # Don't include donor languages.
    return [language for language in languages if language not in donors]


def select_languages(wordlist=None, languages=None, donors=None):
    # Use languages and donors to select a subset of languages from wordlist.
    # Get temporary filename and output to that selecting on languages and donors.
    # Input the temporary file as a wordlist and return the wordlist.
    if not wordlist: return wordlist  # Leave as is.
    languages_ = list()
    if languages:
        if languages == 'all': languages = wordlist.cols
        languages_.extend(languages if isinstance(languages, list) else [languages])
    if donors:
        languages_.extend(donors if isinstance(donors, list) else [donors])
    languages_ = list(set(languages_))
    if not languages_: return wordlist  # No languages selected so leave as is.

    with tempfile.TemporaryDirectory() as tmp:
        file_path = Path(tmp).joinpath('tempwordlist').as_posix()
        wordlist.output('tsv', filename=file_path, subset=True,
                        rows=dict(doculect=" in "+str(languages_)),
                        ignore='all', prettify=False)
        # Now read in again.
        wordlist_ = Wordlist(file_path+'.tsv')
        check_coverage(wordlist_)
        return wordlist_


def check_coverage(wl=None):
    print(f"Wordlist has {wl.width} languages, and {wl.height} concepts in {len(wl)} words.")
    for i in range(200, 0, -1):
        if mutual_coverage_check(wl, i):
            print(f"Minimum mutual coverage is at {i} concept pairs.")
            break
    print(f"Average coverage is at {average_coverage(wl):.2f}")


def compose_wl_from_cldf():
    wl = Wordlist.from_cldf(
        sabor().cldf_dir / "cldf-metadata.json",
        columns=["language_id",
                 "language_family",
                 "concept_name",  # From concept relation, name field.
                 "concept_id",  # From concept relation, id field.
                 "value",
                 "form",
                 "segments",
                 "borrowed",
                 "borrowed_score",
                 "donor_language",
                 "donor_value"],
        namespace=(('language_id', 'language'),
                   ('language_family', 'family'),
                   ("concept_id", "concept"),
                   ('segments', 'tokens'))
    )
    return wl


def compose_wl():
    wl = compose_wl_from_cldf()
    wl.add_entries('concept_name', 'concept_name',
                   lambda x: x.lower(), override=True)
    print(len(wl), wl.cols, wl.height, wl.width)
    return wl

    # Store as temporary file to convert concepts.
    # If not for the concept change, could be done with namespace!

    # with tempfile.TemporaryDirectory() as tmp:
    #     file_path = Path(tmp).joinpath('tempwordlist').as_posix()
    #     wl.output('tsv', filename=file_path, ignore='all', prettify=False)
    #
    #     out_file_path = Path(tmp).joinpath('outwordlist.tsv').as_posix()
    #     with open(out_file_path, 'wt') as wlfile:
    #         wrt = csv.writer(wlfile, delimiter='\t')
    #         # Write out header
    #         wrt.writerow(["language", "family",
    #                       "concept_name", "concept",
    #                       "value", "form", "tokens",
    #                       "borrowed", "borrowed_score",
    #                       "donor_language", "donor_value"])
    #
    #         # Get temp file and write to out file (also temp)
    #         file_path = Path(tmp).joinpath('tempwordlist.tsv').as_posix()
    #         with open(file_path) as file:
    #             rdr = csv.reader(file, delimiter="\t")
    #             header = next(rdr)
    #             print("tmp file:", header)
    #
    #             for row in rdr:
    #                 # fixup possible empty concept id.
    #                 # if not row[4]: row[4] = row[3]
    #                 # Lowercase the concept name.
    #                 row[3] = row[3].lower()
    #
    #                 # add to output file.
    #                 new_row = row[1:]
    #                 wrt.writerow(new_row)
    #
    #     wl = Wordlist(out_file_path)
    #
    #     # fixup concept id and name.
    #
    #     print(len(wl), wl.cols, wl.height, wl.width)
    #     return wl


def get_wordlist(filename=None):
    if filename:
        filepath = Path("foreign").joinpath(filename+".tsv").as_posix()
        print(f"Foreign wordlist - file path: {filepath}", file=sys.stderr)
        wl = Wordlist(filepath)
        # print(f"Cols {wl.columns}", file=sys.stderr)
    else:
        wl = compose_wl()
    return wl


# Get dictionary of family:language from wordlist.
def get_language_family(wl):
    families = {}
    for (ID, language, family) in wl.iter_rows('doculect', 'family'):
        families[language] = family

    return families


# Get list of thresholds from wordlist.
def get_thresholds(cluster_desc):
    tex = re.compile(r"_(\d\.\d+)")
    thresholds = tex.findall(cluster_desc)
    return [float(t) for t in thresholds]


# Enumeration of prediction versus truth for 0, 1 values.
class PredStatus(Enum):
    TN = 1
    TP = 2
    FP = 3
    FN = 4
    F = 5
    T = 6
    NTN = 7
    ALL = 0


#  Functions adapted from pybor for assessment.
def assess_pred(pred, gold, anyPred=None):
    """
    Test 0, 1 prediction versus 0, 1 truth
    """
    if anyPred and gold == 1:
        if pred == 0: return PredStatus.TN

    if pred == gold:
        if gold == 0: return PredStatus.TN
        elif gold == 1: return PredStatus.TP
    else:
        if gold == 0: return PredStatus.FP
        elif gold == 1: return PredStatus.FN
    raise ValueError(f"Pred {pred}, gold {gold}.")


def report_assessment(status, result):
    return (status == PredStatus.ALL or
            (status == PredStatus.F and
             (result == PredStatus.FN or result == PredStatus.FP)) or
            (status == PredStatus.T and
             (result == PredStatus.TN or result == PredStatus.TP)) or
            (status == PredStatus.NTN and result != PredStatus.TN) or
            result == status)


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


def report_results(results, folder, filename):
    file_path = Path(folder).joinpath(filename).as_posix()
    header = ['tp', 'tn', 'fp',  'fn', 'precision', 'recall', 'F1', 'Accuracy']
    words_table = tabulate(results, headers=header, tablefmt="pip", floatfmt=".3f")
    with open(file_path + '.txt', 'w') as f:
        print(words_table, file=f)

    # Report as .tsv
    with open(file_path + '.tsv', 'w', newline='') as f:
        wrt = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        wrt.writerow(header)
        for row in results:
            wrt.writerow(row)


# Functions for reporting help.
#
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


def run(args):
    ...
