"""
Given a constructor function for an analysis method, perform
a k-fold cross validation using pre-established k-fold partition
of train and test files.

Return only overall statistics for each partition, and report
table of statistics along with means and standard deviations.

John E. Miller, May 18, 2022
"""

from functools import partial
import statistics
from tabulate import tabulate
from lingpy import Wordlist, LexStat
from lexibank_sabor import our_path
from lexibank_sabor import (sca_distance, edit_distance)
from saborncommands import (closest, cognate)


# Constructors for use in cross-validation.
#
# To do: See if there is a way to just pass class and dictionary
#        instead of this redundancy for each method.
#
# Constructor for closest match method.
def closest_match_constructor(infile,
                              donors,
                              func,
                              family="language_family",
                              segments="tokens",
                              known_donor="donor_language",
                              **kw):
    return closest.SimpleDonorSearch(
        infile,
        donors=donors,
        func=func,
        family=family,
        segments=segments,
        known_donor=known_donor,
        **kw
    )


# Constructor for cognate based method.
def cognate_based_constructor(infile,
                              donors,
                              func,
                              family="language_family",
                              segments="tokens",
                              # ipa="form",
                              # donor_lng="source_language",
                              # donor_id="source_id",
                              known_donor="donor_language",
                              **kw):
    return cognate.CognateBasedBorrowingDetection(
        infile,
        donors,
        func=func,
        family=family,
        segments=segments,
        known_donor=known_donor,
        **kw
    )


def evaluate_borrowings(wordlist, pred, gold, donors,
                        donor_families, family="family",
                        beta=1.0):
    """
    Return F and related scores for the donor detection.
    """
    # Check for None and be sure of pred versus gold.
    # Return tp, tn, fp, fn, precision, recall, f1score, accuracy.
    # Evaluation wordlist is from predict function of analysis method.
    fn = fp = tn = tp = 0

    for idx, pred_lng, gold_lng in wordlist.iter_rows(pred, gold):
        if wordlist[idx, family] not in donor_families:
            if not pred_lng:
                if not gold_lng: tn += 1
                elif gold_lng in donors: fn += 1
                else: tn += 1
            elif pred_lng:
                if not gold_lng: fp += 1
                elif gold_lng in donors: tp += 1
                else: fp += 1
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = tp/(tp + (fp + fn)/2)
    fb = (1.0 + beta**2.0) * (precision * recall) / \
        (beta**2.0 * precision + recall)
    accuracy = (tp + tn)/(tp + tn + fp + fn)

    return {'fn': fn, 'fp': fp, 'tn': tn, 'tp': tp,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'fb': round(fb, 3),
            'accuracy': round(accuracy, 3)}


def evaluate_fold(constructor, dir, k, fold):
    """
    Evaluate single fold of k-fold train, test datasets,
    using analysis function instantiated by constructor function.
    :param constructor:
    :param dir:
    :param k:
    :param fold:
    :return:
    """
    train_name = "CV{k:d}-fold-{it:02d}-train.tsv".format(k=k, it=fold)
    file_path = our_path(dir, train_name)

    detector = constructor(file_path)
    detector.train(verbose=False)
    # wl = Wordlist(file_path)
    detector.predict_on_wordlist(detector)
    results_train = evaluate_borrowings(
        detector,
        pred="source_language",
        gold=detector.known_donor,
        donors=detector.donors,
        donor_families=detector.donor_families,
        family=detector.family)
    results_train["threshold"] = detector.best_t
    results_train["fold"] = fold
    # print(results_train)

    test_name = "CV{k:d}-fold-{it:02d}-test.tsv".format(k=k, it=fold)
    file_path = our_path(dir, test_name)

    wl = LexStat(file_path) if isinstance(detector, LexStat) \
        else Wordlist(file_path)

    # wl = Wordlist(file_path)
    detector.predict_on_wordlist(wl)
    results_test = evaluate_borrowings(
        wl,
        pred="source_language",
        gold=detector.known_donor,
        donors=detector.donors,
        donor_families=detector.donor_families,
        family=detector.family)

    results_test["threshold"] = detector.best_t
    results_train["fold"] = fold

    return results_test


def evaluate_k_fold(constructor, dir, k):
    """
    Perform k-fold cross-validation using analysis function instantiated by
    constructor function.
    :param constructor:
    :param dir:
    :param k:
    :return:
    """

    cross_val = []
    print("folds: ")
    for fold in range(k):
        results = evaluate_fold(
            constructor, dir=dir, k=k, fold=fold)
        # print(fold, results)
        print(fold, ' ')
        results["fold"] = fold
        cross_val.append(results)
    print()
    means = dict()
    stdevs = dict()
    for key in cross_val[0].keys():
        if key == "fold": continue
        results = [cross_val[fold][key] for fold in range(k)]
        means[key] = statistics.mean(results)
        stdevs[key] = statistics.stdev(results)

    means["fold"] = "mean"
    stdevs["fold"] = "stdev"
    cross_val.append(means)
    cross_val.append(stdevs)

    return cross_val


def register(parser):
    parser.add_argument(
        "k",
        type=int,
        help="Existing k-fold factor to select cross-validation files."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="splits",
        help="Directory to select cross-validation files from."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number to select for a 1-shot cross-validation."
    )
    parser.add_argument(
        "--donors",
        type=str,
        nargs="*",
        default="Spanish",
        help="Donor languages for focused analysis."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cmsca",
        choices=['cmsca', 'cmned', 'cbcbsca', 'cbmtlex'],
        help="Code for borrowing detection method."
    )


def run(args):
    cmsca = partial(closest_match_constructor,
                    func=sca_distance,
                    donors=args.donors)

    cmned = partial(closest_match_constructor,
                    func=edit_distance,
                    donors=args.donors)

    cbcbsca = partial(cognate_based_constructor,
                      func=cognate.cbds_sca,
                      donors=args.donors)
    cbcbsca.keywords['func'].__name__ = 'cognate_based_cognate_sca'

    cbmtlex = partial(cognate_based_constructor,
                      func=cognate.mtbds_lex,
                      donors=args.donors)
    cbmtlex.keywords['func'].__name__ = 'cognate_based_multi_threshold_lexstat'

    methods = {'cmsca': cmsca, 'cmned': cmned,
               'cbcbsca': cbcbsca, 'cbmtlex': cbmtlex}

    constructor = methods[args.method]
    func_name = constructor.keywords['func'].__name__

    if args.fold is not None:
        description = "Cross-validation fold {f} of {k}-folds " \
                      "on {d} directory using {fn}".\
            format(f=args.fold, k=args.k, d=args.dir, fn=func_name)
        args.log.info(description)
        results = evaluate_fold(constructor, dir=args.dir,
                                k=args.k, fold=args.fold)
        print(description)
        print(results)
    else:
        description = "{k}-fold cross-validation " \
                      "on {d} directory using {fn}.".\
            format(k=args.k, d=args.dir, fn=func_name)
        args.log.info(description)
        results = evaluate_k_fold(constructor, dir=args.dir, k=args.k)

        print(description)
        print(tabulate(results, headers='keys',
                       floatfmt=('.1f', '.1f', '.1f', '.1f',
                                 '.3f', '.3f', '.3f', '.3f', '.3f', '.2f')))
