"""
Given a constructor function for an analysis method, perform
a k-fold cross validation using pre-established k-fold partition
of train and test files.

Return only overall statistics for each partition, and report
table of statistics along with means and standard deviations.

"""
import time
from functools import partial
import statistics
from tabulate import tabulate
import unicodedata
import re

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from lingpy import Wordlist
from lexibank_sabor import (
        our_path, evaluate_borrowings,
        get_language_list, subset_wl)
from saborcommands import (closest, cognate, classifier, least)


# Constructors for use in cross-validation.
#
# To do: See if there is a way to just pass class and dictionary
#        instead of this redundancy for each method.
#
# Constructor for closest match method.
def closest_match_constructor(infile,
                              donors,
                              func,
                              segments="tokens",
                              known_donor="donor_language",
                              **kw):
    return closest.SimpleDonorSearch(
        infile,
        donors=donors,
        func=func,
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


# Constructor for least cross-entropy base method
def least_cross_entropy_constructor(infile,
                                    donors,
                                    direction="forward",
                                    approach="dominant",
                                    segments="tokens",
                                    known_donor="donor_language",
                                    **kw):
    return least.LeastCrossEntropy(
        infile,
        donors,
        direction=direction,
        approach=approach,
        segments=segments,
        known_donor=known_donor,
        **kw
    )


# Constructor for classifier based method.
def classifier_based_constructor(infile,
                                 donors,
                                 clf,
                                 funcs,
                                 least_ce,
                                 cognate_cf,
                                 props,
                                 props_tar,
                                 segments="tokens",
                                 known_donor="donor_language",
                                 **kw):
    return classifier.ClassifierBasedBorrowingDetection(
        infile,
        donors,
        clf=clf,
        funcs=funcs,
        least_ce=least_ce,
        cognate_cf=cognate_cf,
        props=props,
        props_tar=props_tar,
        segments=segments,
        known_donor=known_donor,
        **kw
    )


# Language names are normalized within the database split.
def normalize_names(names):
    # strip off accents
    names = [''.join(c for c in unicodedata.normalize('NFD', s)
                     if unicodedata.category(c) != 'Mn')
             for s in names]
    # strip special characters
    names = [re.sub(r'\W+', '', s) for s in names]
    return names


def evaluate_fold(constructor, folder, k, fold, languages=None):
    """
    Evaluate single fold of k-fold train, test datasets,
    using analysis function instantiated by constructor function.
    :param constructor:
    :param folder:
    :param k:
    :param fold:
    :param languages:
    :return:
    """
    train_name = "CV{k:d}-fold-{f:02d}-train.tsv".format(k=k, f=fold)
    file_path = our_path(folder, train_name)

    # Subset language here
    wl = Wordlist(file_path)
    if languages:
        wl = subset_wl(wl, languages)
    detector = constructor(wl)

    # detector = constructor(file_path)
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
            donors=detector.donors)

        if hasattr(detector, "best_key"):
            results[detector.best_key] = round(detector.best_value, 3)
        results["fold"] = fold

        return results

    test_name = "CV{k:d}-fold-{f:02d}-test.tsv".format(k=k, f=fold)
    file_path = our_path(folder, test_name)

    # Subset language here
    wl = Wordlist(file_path)
    if languages:
        wl = subset_wl(wl, languages)
    wl = detector.construct_wordlist(wl)

    results_test = get_results_for_wl(wl)

    return results_test


def evaluate_k_fold(constructor, folder, k, languages=None):
    """
    Perform k-fold cross-validation using analysis function instantiated by
    constructor function.
    :param constructor:
    :param folder:
    :param k:
    :param languages:
    :return:
    """

    cross_val = []
    print("folds: ")
    for fold in range(k):
        results = evaluate_fold(
            constructor, folder=folder, k=k, fold=fold, languages=languages)
        results["fold"] = fold
        print(results)
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
        "--language",
        nargs="*",
        type=str,
        default=None,
        choices=["Yaqui", "Zinacantán Tzotzil", "Q'eqchi'",
                 "Otomi", "Imbabura Quechua", "Wichí",
                 "Mapudungun"],
        help="Subset of languages to include; default is all languages."
    )
    parser.add_argument(
        "--donor",
        type=str,
        nargs="*",
        default=["Spanish"],
        help="Donor languages for focused analysis."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cm_sca",
        choices=['cm_sca', 'cm_ned', 'cm_sca_ov', 'cm_sca_lo',
                 'cb_sca', 'cb_ned',
                 'cb_sca_lo', 'cb_sca_ov',
                 'lce_for', 'lce_back',
                 'lce_for_bor', 'lce_back_bor',
                 'cl_simple', 'cl_rbf_simple',
                 'cl_poly_simple', 'cl_lr_simple',
                 'cl_simple_balanced',
                 'cl_simple_no_props', 'cl_rbf_simple_no_props',
                 'cl_poly_simple_no_props', 'cl_lr_simple_no_props',
                 'cl_ned', 'cl_sca',
                 'cl_all_funcs',
                 'cl_all_funcs_no_props',
                 'cl_least', 'cl_least_no_props',
                 'cl_rbf_least', 'cl_rbf_least_no_props',
                 'cl_lr_least', 'cl_lr_least_no_props',
                 'cl_simple_least',
                 'cl_rbf_simple_least',
                 'cl_lr_simple_least',
                 'cl_simple_least_balanced',
                 'cl_simple_least_no_props',
                 'cl_rbf_simple_least_no_props',
                 'cl_lr_simple_least_no_props',
                 'cl_simple_cognate', 'cl_cognate',
                 'cl_simple_cognate_least',
                 'cl_rbf_simple_cognate_least',
                 'cl_lr_simple_cognate_least',
                 'cl_simple_cognate_least_balanced',
                 'cl_cognate_least',
                 'cl_simple_cognate_least_no_props',
                 'cl_lr_simple_cognate_least_no_props'],
        help="Code for borrowing detection method."
    )


def run(args):

    cm_sca_gl = partial(closest_match_constructor,
                        func=closest.sca_gl,
                        donors=args.donor)
    cm_sca_ov = partial(closest_match_constructor,
                        func=closest.sca_ov,
                        donors=args.donor)
    cm_sca_lo = partial(closest_match_constructor,
                        func=closest.sca_lo,
                        donors=args.donor)

    cm_ned = partial(closest_match_constructor,
                     func=closest.ned,
                     donors=args.donor)

    cb_sca_gl = partial(cognate_based_constructor,
                        func=cognate.cb_sca_gl,
                        donors=args.donor)
    cb_sca_gl.keywords['func'].__name__ = \
        'cognate_based_cognate_sca_global'

    cb_sca_ov = partial(cognate_based_constructor,
                        func=cognate.cb_sca_ov,
                        donors=args.donor)
    cb_sca_ov.keywords['func'].__name__ = \
        'cognate_based_cognate_sca_overlap'

    cb_sca_lo = partial(cognate_based_constructor,
                        func=cognate.cb_sca_lo,
                        donors=args.donor)
    cb_sca_lo.keywords['func'].__name__ = \
        'cognate_based_cognate_sca_local'

    cb_ned = partial(cognate_based_constructor,
                     func=cognate.cb_ned,
                     donors=args.donor)
    cb_ned.keywords['func'].__name__ = \
        'cognate_based_cognate_ned'

    lce_for = partial(least_cross_entropy_constructor,
                      direction="forward",
                      approach="dominant",
                      donors=args.donor)

    lce_back = partial(least_cross_entropy_constructor,
                       direction="backward",
                       approach="dominant",
                       donors=args.donor)
    lce_for_bor = partial(least_cross_entropy_constructor,
                          direction="forward",
                          approach="borrowed",
                          donors=args.donor)

    lce_back_bor = partial(least_cross_entropy_constructor,
                           direction="backward",
                           approach="borrowed",
                           donors=args.donor)

    cl_simple = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple'

    cl_simple_balanced = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear", class_weight='balanced'),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple_balanced.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple_balanced'

    cl_rbf_simple = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_rbf_simple.keywords['func'].__name__ = \
        'classifier_based_rbf_svm_simple'

    cl_poly_simple = partial(
        classifier_based_constructor,
        clf=SVC(kernel="poly"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_poly_simple.keywords['func'].__name__ = \
        'classifier_based_poly_svm_simple'

    cl_lr_simple = partial(
        classifier_based_constructor,
        clf=LogisticRegression(solver='lbfgs', max_iter=1000),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_lr_simple.keywords['func'].__name__ = \
        'classifier_based_lr_simple'

    cl_ned = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_ned.keywords['func'].__name__ = \
        'classifier_based_linear_svm_ned'

    cl_sca = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_sca.keywords['func'].__name__ = \
        'classifier_based_linear_svm_sca'

    cl_all_funcs = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl,
               classifier.clf_sca_lo, classifier.clf_sca_ov],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_all_funcs.keywords['func'].__name__ = \
        'classifier_based_linear_svm_all_functions'

    cl_simple_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_simple_no_props.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple_no_props'

    cl_rbf_simple_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_rbf_simple_no_props.keywords['func'].__name__ = \
        'classifier_based_rbf_svm_simple_no_props'

    cl_poly_simple_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="poly"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_poly_simple_no_props.keywords['func'].__name__ = \
        'classifier_based_poly_svm_simple_no_props'

    cl_lr_simple_no_props = partial(
        classifier_based_constructor,
        clf=LogisticRegression(solver='lbfgs', max_iter=1000),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_lr_simple_no_props.keywords['func'].__name__ = \
        'classifier_based_lr_simple_no_props'

    cl_all_funcs_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl,
               classifier.clf_sca_ov, classifier.clf_sca_lo],
        least_ce=None,
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_all_funcs_no_props.keywords['func'].__name__ = \
        'classifier_based_linear_svm_all_funcs_no_props'

    cl_simple_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple_least.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+least_CE'

    cl_rbf_simple_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_rbf_simple_least.keywords['func'].__name__ = \
        'classifier_based_rbf_svm_simple+least_CE'

    cl_lr_simple_least = partial(
        classifier_based_constructor,
        clf=LogisticRegression(solver='lbfgs', max_iter=1000),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_lr_simple_least.keywords['func'].__name__ = \
        'classifier_based_lr_simple+least_CE'

    cl_simple_least_balanced = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear", class_weight='balanced'),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple_least_balanced.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+least_CE_balanced'

    cl_simple_least_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_simple_least_no_props.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+least_CE_no_props'

    cl_rbf_simple_least_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_rbf_simple_least_no_props.keywords['func'].__name__ = \
        'classifier_based_rbf_svm_simple+least_CE_no_props'

    cl_lr_simple_least_no_props = partial(
        classifier_based_constructor,
        clf=LogisticRegression(solver='lbfgs', max_iter=1000),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_lr_simple_least_no_props.keywords['func'].__name__ = \
        'classifier_based_lr_simple+least_CE_no_props'

    cl_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_least.keywords['func'].__name__ = \
        'classifier_based_linear_svm_least_CE'

    cl_rbf_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_rbf_least.keywords['func'].__name__ = \
        'classifier_based_rbf_svm_least_CE'

    cl_least_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_least_no_props.keywords['func'].__name__ = \
        'classifier_based_linear_svm_least_CE_no_props'

    cl_rbf_least_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[],
        least_ce=['dominant'],
        cognate_cf=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_rbf_least_no_props.keywords['func'].__name__ = \
        'classifier_based_rbf_svm_least_CE_no_props'

    cl_simple_cognate = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=None,
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple_cognate.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+cognate_based'

    cl_cognate = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[],
        least_ce=None,
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_cognate.keywords['func'].__name__ = \
        'classifier_based_linear_svm_cognate_based'

    cl_simple_cognate_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple_cognate_least.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+cognate_based+least_CE'

    cl_rbf_simple_cognate_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="rbf"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_rbf_simple_cognate_least.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+cognate_based+least_CE'

    cl_simple_cognate_least_balanced = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear", class_weight='balanced'),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_simple_cognate_least_balanced.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+cognate_based+least_CE_balanced'


    cl_simple_cognate_least_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_simple_cognate_least_no_props.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+cognate+least_CE_no_props'

    cl_lr_simple_cognate_least = partial(
        classifier_based_constructor,
        clf=LogisticRegression(solver='lbfgs', max_iter=1000),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_lr_simple_cognate_least.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple+cognate_based+least_CE'

    cl_lr_simple_cognate_least_no_props = partial(
        classifier_based_constructor,
        clf=LogisticRegression(solver='lbfgs', max_iter=1000),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_lr_simple_cognate_least_no_props.keywords['func'].__name__ = \
        'classifier_based_lr_simple+cognate+least_CE_no_props'

    cl_cognate_least = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[],
        least_ce=['dominant'],
        cognate_cf='standard',
        props=[],
        props_tar=None,
        donors=args.donor)
    cl_cognate_least.keywords['func'].__name__ = \
        'classifier_based_linear_svm_cognate_based+least_CE'

    methods = {'cm_sca': cm_sca_gl, 'cm_ned': cm_ned,
               'cm_sca_ov': cm_sca_ov, 'cm_sca_lo': cm_sca_lo,
               'cb_sca': cb_sca_gl, 'cb_ned': cb_ned,
               'cb_sca_ov': cb_sca_ov,
               'cb_sca_lo': cb_sca_lo,
               'lce_for': lce_for, 'lce_back': lce_back,
               'lce_for_bor': lce_for_bor, 'lce_back_bor': lce_back_bor,
               'cl_simple': cl_simple,
               'cl_rbf_simple': cl_rbf_simple,
               'cl_poly_simple': cl_poly_simple,
               'cl_lr_simple': cl_lr_simple,
               'cl_simple_balanced': cl_simple_balanced,
               'cl_ned': cl_ned, 'cl_sca': cl_sca,
               'cl_all_funcs': cl_all_funcs,
               'cl_simple_no_props': cl_simple_no_props,
               'cl_rbf_simple_no_props': cl_rbf_simple_no_props,
               'cl_poly_simple_no_props': cl_poly_simple_no_props,
               'cl_lr_simple_no_props': cl_lr_simple_no_props,
               'cl_all_funcs_no_props': cl_all_funcs_no_props,
               'cl_simple_least': cl_simple_least,
               'cl_rbf_simple_least': cl_rbf_simple_least,
               'cl_lr_simple_least': cl_lr_simple_least,
               'cl_simple_least_balanced': cl_simple_least_balanced,
               'cl_simple_least_no_props': cl_simple_least_no_props,
               'cl_rbf_simple_least_no_props': cl_rbf_simple_least_no_props,
               'cl_lr_simple_least_no_props': cl_lr_simple_least_no_props,
               'cl_least': cl_least,
               'cl_rbf_least': cl_rbf_least,
               'cl_least_no_props': cl_least_no_props,
               'cl_rbf_least_no_props': cl_rbf_least_no_props,
               'cl_simple_cognate': cl_simple_cognate,
               'cl_cognate': cl_cognate,
               'cl_simple_cognate_least': cl_simple_cognate_least,
               'cl_rbf_simple_cognate_least': cl_rbf_simple_cognate_least,
               'cl_lr_simple_cognate_least': cl_lr_simple_cognate_least,
               'cl_simple_cognate_least_balanced': cl_simple_cognate_least_balanced,
               'cl_cognate_least': cl_cognate_least,
               'cl_simple_cognate_least_no_props': cl_simple_cognate_least_no_props,
               'cl_lr_simple_cognate_least_no_props': cl_lr_simple_cognate_least_no_props
               }

    start_time = time.time()

    constructor = methods[args.method]
    if fn := constructor.keywords.get('func'):
        func_name = fn.__name__
    else:
        func_name = args.method

    if args.language:
        args.language = get_language_list(args.language, args.donor)
        args.log.info("Languages: {l}, donors: {d}".
                      format(l=args.language, d=args.donor))
        language_ids = normalize_names(args.language)
    else:
        language_ids = None

    if args.fold is not None:
        description = "Cross-validation fold {f} of {k}-folds " \
                      "on {d} directory using {fn}".\
            format(f=args.fold, k=args.k, d=args.dir, fn=func_name)
        args.log.info(description)
        results = evaluate_fold(constructor, folder=args.dir,
                                k=args.k, fold=args.fold,
                                languages=language_ids)
        print(description)
        if args.language:
            print("Languages: {l}, donors: {d}".
                  format(l=args.language, d=args.donor))
        print(results)
    else:
        description = "{k}-fold cross-validation " \
                      "on {d} directory using {fn}.".\
            format(k=args.k, d=args.dir, fn=func_name)
        args.log.info(description)
        results = evaluate_k_fold(constructor, folder=args.dir,
                                  k=args.k, languages=language_ids)

        print(description)
        if args.language:
            print("Languages: {l}, donors: {d}".
                  format(l=args.language, d=args.donor))
        print(tabulate(results, headers='keys',
                       floatfmt=('.1f', '.1f', '.1f', '.1f',
                                 '.3f', '.3f', '.3f', '.3f', '.3f', '.2f')))

    # print("--- %s seconds ---" % (time.time() - start_time))
    secs = time.time() - start_time
    print(f'--- seconds --- {int(secs) // 3600:02}:'
          f'{(int(secs) % 3600) // 60:02}:{secs % 60:02.2f}')
