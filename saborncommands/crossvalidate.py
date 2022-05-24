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
from sklearn.svm import SVC
from lingpy import Wordlist, LexStat
from lexibank_sabor import (our_path, evaluate_borrowings)
from lexibank_sabor import (sca_distance, edit_distance)
from saborncommands import (closest, cognate, classifier)


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


# Constructor for classifier based method.
def classifier_based_constructor(infile,
                                 donors,
                                 clf,
                                 funcs,
                                 props,
                                 family="language_family",
                                 segments="tokens",
                                 known_donor="donor_language",
                                 **kw):
    return classifier.ClassifierBasedBorrowingDetection(
        infile,
        donors,
        clf=clf,
        funcs=funcs,
        props=props,
        family=family,
        segments=segments,
        known_donor=known_donor,
        **kw
    )


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

    def get_results_for_wl(wl_):
        ln_in = len(wl_)
        detector.predict_on_wordlist(wl_)
        ln_out = len(wl_)
        if ln_in != ln_out:
            print("*** In {} and out {} wordlist lengths different".
                  format(ln_in, ln_out))
        results = evaluate_borrowings(
            wl_,
            pred="source_language",
            gold=detector.known_donor,
            donors=detector.donors,
            donor_families=detector.donor_families,
            family=detector.family)

        if hasattr(detector, "best_key"):
            results[detector.best_key] = round(detector.best_value, 3)
        results["fold"] = fold

        return results

    test_name = "CV{k:d}-fold-{it:02d}-test.tsv".format(k=k, it=fold)
    file_path = our_path(dir, test_name)

    wl = LexStat(file_path) if isinstance(detector, LexStat) \
        else Wordlist(file_path)

    results_test = get_results_for_wl(wl)

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
        choices=['cmsca', 'cmned', 'cbsca', 'cblex', 'clsvcdist'],
        help="Code for borrowing detection method."
    )


def run(args):
    cmsca = partial(closest_match_constructor,
                    func=sca_distance,
                    donors=args.donors)

    cmned = partial(closest_match_constructor,
                    func=edit_distance,
                    donors=args.donors)

    cbsca = partial(cognate_based_constructor,
                    func=cognate.cbds_sca,
                    donors=args.donors,
                    runs=None)
    cbsca.keywords['func'].__name__ = 'cognate_based_cognate_sca'

    cblex = partial(cognate_based_constructor,
                    func=cognate.cbds_lex,
                    donors=args.donors,
                    runs=1000,
                    lexstat=True)
    cblex.keywords['func'].__name__ = 'cognate_based_cognate_lexstat'

    clsvcdist = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_sca, classifier.clf_ned],
        props=None,
        props_tar=None,
        by_tar=False,
        donors=args.donors,
        family="language_family")
    clsvcdist.keywords['func'].__name__ = \
        'classifier_based_SVM_linear_sca_ned'

    methods = {'cmsca': cmsca, 'cmned': cmned,
               'cbsca': cbsca, 'cblex': cblex,
               'clsvcdist': clsvcdist}

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
