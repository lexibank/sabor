"""
Borrowings by donor words detected as cognates.
"""

from lingpy import *
from lexibank_sabor import (
        get_our_wordlist,  # sca_distance, edit_distance,
        evaluate_borrowings_fs,
        our_path,
    )
from functools import partial
from numpy.polynomial import Polynomial
from tabulate import tabulate
import lingrex
from lingrex import borrowing


def multi_threshold_based_donor_search(
        wl,
        donors=None,
        family="language_family",
        donor_lng="source_language",
        donor_id="source_id",
        method='sca',
        model='sca',
        threshold=None,
        int_threshold=None,
        runs=5000,  # Only with lexstat
        mode="overlap",
        cluster_method="upgma",
        ):
    """
    Perform two threshold clustering on wl configured according to arguments.

    :param wl:
    :param donors:
    :param family:
    :param donor_lng:
    :param donor_id:
    :param method:
    :param model:
    :param threshold:
    :param runs:
    :param int_threshold:
    :param mode:
    :param cluster_method:
    """
    # See paper, section "4 Results" and section "3.2 Methods".

    cog_id = "cog_id"
    if cog_id not in wl.columns:
        lingrex.borrowing.internal_cognates(
            wl,
            family=family,
            partial=False,
            runs=runs,
            ref=cog_id,
            method=method,
            threshold=int_threshold,
            cluster_method=cluster_method,
            model=model)

    bor_id = "bor_id"
    # Can't override name use in wordlist with LingRex.
    # So number bor_ids in sequence.
    next_id = len([b for b in wl.columns if b.startswith(bor_id)])
    bor_id_ = bor_id + "{:d}".format(next_id)
    lingrex.borrowing.external_cognates(
        wl,
        cognates=cog_id,
        family=family,
        ref=bor_id_,
        threshold=threshold,
        align_mode=mode)

    get_wl_borrowings(wl, donors, family, donor_id, donor_lng, bor_id_)


# Cluster function - invoked in training.
# Use partial functions to pre-specify standard configurations.
def cognate_based_donor_search(
        wl,
        donors=None,
        family="language_family",
        donor_lng="source_language",
        donor_id="source_id",
        method='sca',
        model='sca',
        threshold=None,
        runs=5000,  # Only with lexstat
        mode="overlap",
        cluster_method="upgma",
        ):
    """
    Perform clustering on wl configured according to arguments.

    :param wl:
    :param donors:
    :param family:
    :param donor_id:
    :param donor_lng:
    :param method:
    :param model:
    :param threshold:
    :param runs:
    :param mode:
    :param cluster_method:
    """

    if method == "lexstat":
        wl.get_scorer(runs=runs)

    bor_id = "bor_id"
    wl.cluster(method=method,
               model=model,
               threshold=threshold,
               mode=mode,
               cluster_method=cluster_method,
               ref=bor_id,
               override=True)

    get_wl_borrowings(wl, donors, family, donor_id, donor_lng, bor_id)


def get_wl_borrowings(wl, donors, family, donor_id, donor_lng, bor_id):
    donors = [donors] if isinstance(donors, str) else donors

    # Add donor language referents.
    wl.add_entries(donor_lng, bor_id, lambda x: "", override=True)
    wl.add_entries(donor_id, bor_id, lambda x: "", override=True)
    # Get dictionary representations of cluster ids.
    etd = wl.get_etymdict(ref=bor_id)
    # Zero out ids that do not cross families or donor.
    for cog_id, values in etd.items():
        if cog_id == 0: continue  # Skip 0 BorIds

        indices = []
        for v in values:
            if v: indices += v
        families = [wl[idx, family] for idx in indices]
        # If set of just 1 family then local cognate.
        if len(set(families)) == 1:
            for idx in indices: wl[idx, bor_id] = 0
            continue

        # Add donor_lng and donor_id (first encountered)
        loans = [
            (donor, idx) for idx in indices
            for donor in donors
            if wl[idx, 'doculect'].startswith(donor)]
        if len(loans) == 0:
            for idx in indices: wl[idx, bor_id] = 0
            continue

        donor_, idx_ = loans[0]  # Take first even if more than one.
        # Might want to add shortest distance instead.
        for idx in indices:
            if not any(wl[idx, 'doculect'].startswith(donor)
                       for donor in donors):
                wl[idx, donor_lng] = donor_
                wl[idx, donor_id] = idx_
                
    # Wordlist now has borrowing indices and within concept cluster indices.
    # Set donor languages and donor indices for this threshold.


cbds_sca = partial(cognate_based_donor_search,
                   method='sca', model='sca',
                   runs=None, mode='global',
                   cluster_method='upgma')

# Could increase runs and change cluster_method if useful.
cbds_lex = partial(cognate_based_donor_search,
                   method='lexstat', model='sca',
                   runs=2000, mode='overlap',
                   cluster_method='infomap')

mtbds_sca = partial(multi_threshold_based_donor_search,
                    method='sca', model='sca',
                    runs=None, mode="global",
                    cluster_method="upgma",
                    int_threshold=0.30)

# Could increase runs, change cluster_method, and int_threshold if useful.
mtbds_lex = partial(multi_threshold_based_donor_search,
                    method='lexstat', model='sca',
                    runs=2000, mode="overlap",
                    cluster_method="upgma",
                    int_threshold=0.50)


# Early version of init just for cognate based.
# Subsequent could include the cognate based function as argument.
# e.g., cgf=None  # cognate based detection function.
class CognateBasedBorrowingDetection(LexStat):

    def __init__(
            self,
            infile,
            donors,
            func=None,
            family="language_family",
            segments="tokens",
            ipa="form",
            donor_lng="source_language",
            donor_id="source_id",
            known_donor="donor_language",
            **kw
            ):
        """

        :param self:
        :param infile:
        :param donors:
        :param family:
        :param segments:
        :param known_donor:
        :param kw:
        :return:
        """
        LexStat.__init__(self, infile, ipa=ipa, segments=segments, **kw)

        self.func = func or cbds_sca
        self.donors = [donors] if isinstance(donors, str) else donors
        # Define wordlist field names.
        self.family = family
        self.segments = segments
        self.donor_lng = donor_lng
        self.donor_id = donor_id
        self.known_donor = known_donor
        self.donor_families = {fam for (ID, lang, fam)
                               in self.iter_rows('doculect', self.family)
                               if lang in self.donors}
        self.best_threshold = 0.0
        self.best_f1score = 0.0

    def trial_threshold(self, threshold):
        self.func(
            self,
            donors=self.donors,
            family=self.family,
            donor_lng=self.donor_lng,
            donor_id=self.donor_id,
            threshold=threshold
        )
        fs = evaluate_borrowings_fs(
            self,
            pred=self.donor_lng,
            gold=self.known_donor,
            donors=self.donors,
            donor_families=self.donor_families,
            family=self.family
        )
        return fs

    @staticmethod
    def get_optimal_threshold_(results):
        """
        Analyze threshold trials, calculate and return optimal threshold and
        f1 score values.

        :param results: Table of results from threshold trials.
        :return: Optimal threshold and f1 score values.
        """
        results = [[result[1], result[2]] for result in results]
        results = sorted(results, key=lambda thr_sc: thr_sc[1], reverse=True)
        if len(results) <= 3:  # Choose maximum f1 score.
            return results[0][0], results[0][1]

        poly = Polynomial.fit(x=[row[0] for row in results[:4]],
                              y=[row[1] for row in results[:4]],
                              deg=3)
        thresholds, f1scores = poly.linspace(n=16)
        opt_thr, opt_fs = sorted(zip(thresholds, f1scores),
                                 key=lambda result: result[1],
                                 reverse=True)[0]
        return round(opt_thr, 3), round(opt_fs, 3)

    def train(self, thresholds=None, verbose=False):
        """
        Train thresholds on current data to find optimum.

        :param thresholds:
        :param verbose:
        """
        thresholds = thresholds or [i*0.1 for i in range(1, 10)]

        best_idx, best_t, best_fs = 0, 0.0, 0.0
        results = []
        for i, threshold in enumerate(thresholds):
            if verbose: print("cognate threshold {0:.2f}".format(threshold))

            fs = self.trial_threshold(threshold=threshold)
            results += [[i, threshold, fs]]

            if verbose: print("threshold {:.2f}, f1 score {:.3f}".
                              format(threshold, fs))
            best_t, best_fs = self.get_optimal_threshold_(results)

        if verbose:
            print("* Training Results *")
            print(tabulate(results,
                           headers=["index", "threshold", "F1 score"]))

        # Run best threshold for final state of wordlist fs.
        if len(thresholds) != 1:  # Only 1 threshold.
            best_fs = self.trial_threshold(threshold=best_t)
        if verbose: print("Best threshold {:.3f}, f1 score {:.3f}".
                          format(best_t, best_fs))

        self.best_threshold = best_t
        self.best_f1score = best_fs

    def predict(self, donors, targets, wordlist, verbose=False):
        ...

    def predict_on_wordlist(self, wordlist, verbose=False):
        cognate_based_donor_search(
            wl=wordlist,
            donors=self.donors,
            family=self.family,
            donor_lng=self.donor_lng,
            donor_id=self.donor_id,
            threshold=self.best_threshold
        )
        if verbose: print("evaluation at threshold", self.best_threshold)


def run(args):
    # wl = get_our_wordlist()
    # print(wl.height, wl.width, len(wl), wl.columns)
    # bor = CognateBasedBorrowingDetection(
    #     wl, donors="Spanish", family="language_family")
    # args.log.info("loaded wordlist.")
    # print(bor.height, bor.width, len(bor), bor.columns)

    # wl = Wordlist(our_path("splits", "CV10-fold-00-train.tsv"))
    # bor = CognateBasedBorrowingDetection(
    #     wl, donors="Spanish", family="language_family")
    # args.log.info("loaded wordlist.")
    # print(bor.height, bor.width, len(bor), bor.columns)

    # Test cluster function with wl.
    # wl = LexStat(wl)
    # cognate_based_donor_search(wl, donors="Spanish",
    #                            t_idx=1, threshold=0.3,
    #                            family="language_family")
    # wl.output('tsv', filename=our_path("store", "test-cognate-fn"),
    #           prettify=False, ignore="all")

    # Test train and predict_on_wordlist
    # wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
    # bor = CognateBasedBorrowingDetection(
    #     wl, donors="Spanish", family="language_family")
    # results = bor.train(thresholds=[0.5, 0.6, 0.7, 0.8], verbose=True)
    # bor.output(
    #     'tsv',
    #     filename=our_path("store", "CL-train-CV10-fold-00-train"),
    #     prettify=False, ignore="all")
    #
    # wl = LexStat('splits/CV10-fold-00-train.tsv')
    # bor.predict_on_wordlist(wl)
    # wl.output(
    #     'tsv',
    #     filename=our_path("store", "CL-predict-CV10-fold-00-train"),
    #     prettify=False, ignore="all")
    #
    # wl = LexStat('splits/CV10-fold-00-test.tsv')
    # bor.predict_on_wordlist(wl)
    # wl.output(
    #     'tsv',
    #     filename=our_path("store", "CL-predict-CV10-fold-00-test"),
    #     prettify=False, ignore="all")
    #
    # print("* Training Results *")
    # results += [[bor.best_idx, bor.best_threshold, bor.best_f1score]]
    # print(tabulate(results, headers=["iter", "threshold", "F1 score"]))

    # Test Partial
    # wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
    # bor = CognateBasedBorrowingDetection(
    #     wl, func=cbds_sca, donors="Spanish", family="language_family")
    # bor.train(thresholds=[0.4], verbose=True)
    # bor.output(
    #     'tsv',
    #     filename=our_path("store", "CL-partial-CV10-fold-00-train"),
    #     prettify=False, ignore="all")

    # wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
    # bor = CognateBasedBorrowingDetection(
    #     wl, func=cbds_sca, donors="Spanish", family="language_family")
    # bor.train(thresholds=[0.4, 0.5], verbose=True)
    # bor.output(
    #     'tsv',
    #     filename=our_path("store", "CL-partial-4-5-CV10-fold-00-train"),
    #     prettify=False, ignore="all")

    # wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
    # bor = CognateBasedBorrowingDetection(
    #     wl, func=cbds_sca, donors="Spanish", family="language_family")
    # bor.train(verbose=True)
    # bor.output(
    #     'tsv',
    #     filename=our_path("store", "CL-partial-sca-all-CV10-fold-00-train"),
    #     prettify=False, ignore="all")

    # Multi-threshold based borrowing detection.
    wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
    bor = CognateBasedBorrowingDetection(
        wl, func=mtbds_sca, donors="Spanish", family="language_family")
    bor.train(verbose=True)
    bor.output(
        'tsv',
        filename=our_path("store", "CL-partial-sca-mtbds-CV10-fold-00-train"),
        prettify=False, ignore="all")
