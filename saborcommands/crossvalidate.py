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

from lexibank_sabor import (our_path, evaluate_borrowings)
from lexibank_sabor import (sca_distance, edit_distance)
from saborcommands import (closest, cognate, classifier)


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


# Constructor for classifier based method.
def classifier_based_constructor(infile,
                                 donors,
                                 clf,
                                 funcs,
                                 meths,
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
        meths=meths,
        props=props,
        props_tar=props_tar,
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
            donors=detector.donors)

        if hasattr(detector, "best_key"):
            results[detector.best_key] = round(detector.best_value, 3)
        results["fold"] = fold

        return results

    test_name = "CV{k:d}-fold-{it:02d}-test.tsv".format(k=k, it=fold)
    file_path = our_path(dir, test_name)
    wl = detector.construct_wordlist(file_path)

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
                 'cb_sca', 'cb_ned', 'cb_lex',
                 'cb_sca_lo', 'cb_sca_ov', 'cb_lex_lo', 'cb_lex_ov',
                 'cl_full', 'cl_lean',
                 'cl_fast', 'cl_simple',
                 'cl_ned', 'cl_sca',
                 'cl_simplest', 'cl_all_funcs'],
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

    cb_lex_gl = partial(cognate_based_constructor,
                        func=cognate.cb_lex_gl,
                        donors=args.donor,
                        lexstat=True,
                        runs=5000)
    cb_lex_gl.keywords['func'].__name__ = \
        'cognate_based_cognate_lex_global'

    cb_lex_ov = partial(cognate_based_constructor,
                        func=cognate.cb_lex_ov,
                        donors=args.donor,
                        lexstat=True,
                        runs=5000)
    cb_lex_ov.keywords['func'].__name__ = \
        'cognate_based_cognate_lex_overlap'

    cb_lex_lo = partial(cognate_based_constructor,
                        func=cognate.cb_lex_lo,
                        donors=args.donor,
                        lexstat=True,
                        runs=5000)
    cb_lex_lo.keywords['func'].__name__ = \
        'cognate_based_cognate_lex_local'

    cl_full = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_sca_gl, classifier.clf_ned],
        meths=["sca", "lex"],
        props=None,
        props_tar=None,
        donors=args.donor)
    cl_full.keywords['func'].__name__ = \
        'classifier_based_linear_svm_full'

    cl_fast = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned],
        meths=["sca"],
        props=None,
        props_tar=None,
        donors=args.donor)
    cl_fast.keywords['func'].__name__ = \
        'classifier_based_linear_svm_fast'

    cl_simple = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        meths=None,
        props=None,
        props_tar=None,
        donors=args.donor)
    cl_simple.keywords['func'].__name__ = \
        'classifier_based_linear_svm_simple'

    cl_ned = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned],
        meths=None,
        props=None,
        props_tar=None,
        donors=args.donor)
    cl_ned.keywords['func'].__name__ = \
        'classifier_based_linear_svm_ned'

    cl_sca = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_sca_gl],
        meths=None,
        props=None,
        props_tar=None,
        donors=args.donor)
    cl_sca.keywords['func'].__name__ = \
        'classifier_based_linear_svm_sca'

    cl_no_props = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl],
        meths=None,
        props=[],
        props_tar=[],
        donors=args.donor)
    cl_no_props.keywords['func'].__name__ = \
        'classifier_based_linear_svm_no_properties'

    cl_all_funcs = partial(
        classifier_based_constructor,
        clf=SVC(kernel="linear"),
        func=lambda x: x,  # Artificial argument for name.
        funcs=[classifier.clf_ned, classifier.clf_sca_gl,
               classifier.clf_sca_lo, classifier.clf_sca_ov],
        meths=None,
        props=None,
        props_tar=None,
        donors=args.donor)
    cl_all_funcs.keywords['func'].__name__ = \
        'classifier_based_linear_svm_all_functions'

    methods = {'cm_sca': cm_sca_gl, 'cm_ned': cm_ned,
               'cm_sca_ov': cm_sca_ov, 'cm_sca_lo': cm_sca_lo,
               'cb_sca': cb_sca_gl, 'cb_ned': cb_ned, 'cb_lex': cb_lex_gl,
               'cb_sca_ov': cb_sca_ov, 'cb_lex_ov': cb_lex_ov,
               'cb_sca_lo': cb_sca_lo, 'cb_lex_lo': cb_lex_lo,

               'cl_full': cl_full, 'cl_fast': cl_fast,
               'cl_simple': cl_simple,
               'cl_ned': cl_ned, 'cl_sca': cl_sca,
               'cl_no_props': cl_no_props,
               'cl_all_funcs': cl_all_funcs}

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
