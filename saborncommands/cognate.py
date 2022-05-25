"""
Borrowings by donor words detected as cognates.
"""

from lingpy import *
from lexibank_sabor import (
        get_our_wordlist,
        evaluate_borrowings_fs,
        our_path,
    )
from functools import partial
from numpy.polynomial import Polynomial
from tabulate import tabulate
import lingrex
from lingrex import borrowing


def form_lexstat(donors, targets, families):
    """
    Function applies cognate-based or multi-threshold cognate-based
    borrowing detection and returns a dictionary with per item results.
    :param donors:
    :type donors:
    :param families:
    :param targets:
    :type targets:
    :return: simple wordlist.
    """

    # Construct wordlist: dictionary with consecutive integers as keys
    # and lists as values with the key 0 specifying the header.
    wl = {0: ['ID', 'DOCULECT', 'LANGUAGE_FAMILY', 'CONCEPT', 'TOKENS']}
    id_ = 1
    for doculect, segments in donors.items():
        fam = families[doculect] if doculect in families else doculect + "_fam"
        wl[id_] = [id_, doculect, fam, 'undefined', segments]
        id_ += 1
    for doculect, segments in targets.items():
        fam = families[doculect] if doculect in families else doculect + "_fam"
        wl[id_] = [id_, doculect, fam, 'undefined', segments]
        id_ += 1

    return LexStat(wl)


def multi_threshold_based_donor_search(
        wl,
        donors=None,
        family="language_family",
        donor_lng="source_language",
        donor_id="source_id",
        model='sca',
        threshold=None,
        int_threshold=None,
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
    :param model:
    :param threshold:
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
            ref=cog_id,
            method="lexstat",
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
    :param mode:
    :param cluster_method:
    """

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
        # for future case where there are multiple donors.
        for idx in indices:
            if not any(wl[idx, 'doculect'].startswith(donor)
                       for donor in donors):
                wl[idx, donor_lng] = donor_
                wl[idx, donor_id] = idx_
                
    # Wordlist now has borrowing indices and within concept cluster indices.
    # Set donor languages and donor indices for this threshold.


cbds_sca = partial(cognate_based_donor_search,
                   method='sca',
                   mode='global',
                   cluster_method='upgma')

cbds_lex = partial(cognate_based_donor_search,
                   method='lexstat',
                   mode='overlap',
                   cluster_method='upgma')


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
            runs=10000,
            lexstat=False,
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
        if lexstat:
            self.get_scorer(runs=runs)

        self.runs = runs
        self.func = func or cbds_sca
        self.donors = [donors] if isinstance(donors, str) else donors
        # Define wordlist field names.
        self.family = family
        self.segments = segments
        self.lexstat = lexstat
        self.donor_lng = donor_lng
        self.donor_id = donor_id
        self.known_donor = known_donor
        self.donor_families = {fam for (ID, lang, fam)
                               in self.iter_rows('doculect', self.family)
                               if lang in self.donors}
        self.best_value = 0
        self.best_score = 0
        self.best_key = 'threshold'

    def trial_threshold(self, threshold=0.3, donor_lng=None, donor_idx=None):
        self.func(
            self,
            donors=self.donors,
            family=self.family,
            donor_lng=donor_lng,
            donor_id=donor_idx,
            threshold=threshold
        )
        fs = evaluate_borrowings_fs(
            self,
            pred=donor_lng,
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
            donor_lng = "trial_{0}".format(i)
            donor_idx = "idx_{0}".format(i)

            fs = self.trial_threshold(threshold=threshold, donor_lng=donor_lng,
                                      donor_idx=donor_idx)
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
            best_fs = self.trial_threshold(threshold=best_t,
                                           donor_lng=self.donor_lng,
                                           donor_idx=self.donor_id)

        if verbose:
            print("Best threshold {:.3f}, F1 score {:.3f}".
                  format(best_t, best_fs))

        self.best_value = best_t
        self.best_score = best_fs

    def predict(self, donors, targets, families, verbose=False):
        """
        Predict borrowings for one concept for
        given function and trained threshold setting.
        :param donors:
        :param targets:
        :param verbose:
        :param families:
        :return:
        """
        wl = form_lexstat(donors, targets, families)
        self.predict_on_wordlist(wl, verbose=verbose)

        for id_, target, segments, donor, donor_id in wl.iter_rows(
                'doculect', self.segments, self.donor_lng, self.donor_id):
            if donor:
                print("{t}: {d}, {s}: {ds}".format(
                    t=target, d=donor, s=segments,
                    ds=wl[donor_id, self.segments]))

    def predict_on_wordlist(self, wordlist, verbose=False):
        """
        Predict for an entire wordlist using trained threshold setting.
        :param wordlist:
        :param verbose:
        """

        # if self.lexstat:
        if not hasattr(self, "bscorer"):
            wordlist.get_scorer(runs=self.runs)

        cognate_based_donor_search(
            wl=wordlist,
            donors=self.donors,
            family=self.family,
            donor_lng=self.donor_lng,
            donor_id=self.donor_id,
            threshold=self.best_value
        )
        if verbose: print("evaluation at threshold", self.best_value)


class MultiThreshold(CognateBasedBorrowingDetection):

    def train(self, thresholds=None, verbose=False):
        thresholds = [(i*0.1, j*0.1) for i in range(1, 10) for j in range(1, 10)]
        best_idx, best_t, best_fs = 0, 0.0, 0.0
        # WE NEED TO EXTEND THIS LATER AND BREAK OUT STUFF FROM LINGREX TO WORK
        # WELL HERE


def register(parser):
    parser.add_argument(
        "--method",
        type=str,
        default="sca",
        choices=['sca', 'lexstat'],
        help="Choose between sca or lexstat methods."
    )


def run(args):
    wl = get_our_wordlist()

    if args.method == "lexstat":
        # Test predict borrowing for concept.
        wl_ = LexStat(wl)
        bor = CognateBasedBorrowingDetection(
            wl_, func=cbds_lex, runs=5000, lexstat=True,
            donors="Spanish", family="language_family")
        bor.train(thresholds=[0.5, 0.6, 0.7, 0.8], verbose=True)
        bor.predict_on_wordlist(bor, verbose=True)
        bor.output(
            'tsv',
            filename=our_path("store", "CB-predict-all-lex"),
            prettify=False, ignore="all")

        # Test train and predict_on_wordlist
        # wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
        # bor = CognateBasedBorrowingDetection(
        #     wl, func=cbds_lex, runs=2000, lexstat=True,
        #     donors="Spanish", family="language_family")
        # bor.train(thresholds=[0.5, 0.6, 0.7, 0.8], verbose=True)
        # bor.output(
        #     'tsv',
        #     filename=our_path("store", "CB-train-CV10-fold-00-train-lex"),
        #     prettify=False, ignore="all")
        #
        # wl = LexStat('splits/CV10-fold-00-test.tsv')
        # bor.predict_on_wordlist(wl, verbose=True)
        # wl.output(
        #     'tsv',
        #     filename=our_path("store", "CB-predict-CV10-fold-00-test-lex"),
        #     prettify=False, ignore="all")

    else:  # sca
        # Test predict borrowing for concept.
        # wl = LexStat(wl)
        # bor = CognateBasedBorrowingDetection(
        #     wl, func=cbds_sca,
        #     donors="Spanish", family="language_family")
        # bor.train(thresholds=None, verbose=True)
        # wl2 = LexStat(wl)
        # bor.predict_on_wordlist(wl2)
        wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
        bor = CognateBasedBorrowingDetection(
            wl, func=cbds_sca,
            donors="Spanish", family="language_family")
        bor.train(thresholds=[0.3, 0.4, 0.5, 0.6], verbose=True)
        bor.output(
            'tsv',
            filename=our_path("store", "CB-train-CV10-fold-00-train-sca"),
            prettify=False, ignore="all")

        wl = LexStat('splits/CV10-fold-00-test.tsv')
        bor.predict_on_wordlist(wl, verbose=True)
        wl.output(
            'tsv',
            filename=our_path("store", "CB-predict-CV10-fold-00-test-sca"),
            prettify=False, ignore="all")
