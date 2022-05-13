"""
Borrowings by donor words detected as cognates.
"""

from lingpy import *
from lexibank_sabor import (
        get_our_wordlist, sca_distance, edit_distance,
        evaluate_borrowings_fs,
        our_path,
    )
from functools import partial
from tabulate import tabulate


# distance functions - may be useful for distance output to meta method.
def distance_by_idx(idxA, idxB, wordlist, distance=None, **kw):
    return distance(
            wordlist[idxA, wordlist.segments],
            wordlist[idxB, wordlist.segments],
            **kw)


clf_sca = partial(
        distance_by_idx,
        distance=sca_distance)
clf_ned = partial(
        distance_by_idx,
        distance=edit_distance)


# Cluster function - invoked in training.
def cognate_based_donor_search(
        wl,
        donors=None,
        family="language_family",
        donor_lng="source_language",
        donor_id="source_id",
        method='lexstat',
        model='sca',
        t_idx=0,
        threshold=None,
        runs=2000,
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
    :param t_idx:
    :param threshold:
    :param runs:
    :param mode:
    :param cluster_method:
    """

    donors = [donors] if isinstance(donors, str) else donors

    if method == "lexstat":
        wl.get_scorer(runs=runs)

    # Cluster using specified method and threshold
    bor_id = "bor_id_{0}".format(t_idx)

    wl.cluster(method=method,
               model=model,
               threshold=threshold,
               mode=mode,
               cluster_method=cluster_method,
               ref=bor_id)

    # Add entries for cluster id combined with language family.
    wl.add_entries("bor_fam_{0}".format(t_idx), bor_id + "," + family,
                   lambda x, y: str(x[y[0]]) + "-" + x[y[1]])

    # Add donor language referents.
    wl.add_entries(donor_lng, bor_id, lambda x: "")
    wl.add_entries(donor_id, bor_id, lambda x: "")

    # Get dictionary representations of cluster ids.
    etd = wl.get_etymdict(ref=bor_id)
    # Zero out ids that do not cross families.
    # Zero out ids that do not include a donor.
    for cog_id, values in etd.items():
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


# Early version of init just for cognate based.
# Subsequent could include the cognate based function as argument.
# e.g., cgf=None  # cognate based detection function.
class CognateBasedBorrowingDetection(LexStat):

    def __init__(
            self,
            infile,
            donors,
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
        self.best_idx = 0

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
            donor_lng = self.donor_lng + "_{:d}".format(i+1)
            donor_id = self.donor_id + "_{:d}".format(i+1)
            cognate_based_donor_search(
                self,
                donors=self.donors,
                family=self.family,
                donor_lng=donor_lng,
                donor_id=donor_id,
                t_idx=i+1,
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
            if fs > best_fs:
                best_idx = i+1
                best_t = threshold
                best_fs = fs
            if verbose: print("threshold {:.2f}, f1 score {:.3f}".
                              format(threshold, fs))
            results += [[i+1, threshold, fs]]

        self.best_threshold = best_t
        self.best_f1score = best_fs
        self.best_idx = best_idx

        return results

    def predict(self, donors, targets, wordlist):
        ...

    def predict_on_wordlist(self, wordlist):
        cognate_based_donor_search(
            wl=wordlist,
            donors=self.donors,
            family=self.family,
            donor_lng=self.donor_lng,
            donor_id=self.donor_id,
            threshold=self.best_threshold
        )
        print("evaluation at threshold", self.best_threshold)


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

    wl = LexStat(our_path("splits", "CV10-fold-00-train.tsv"))
    bor = CognateBasedBorrowingDetection(
        wl, donors="Spanish", family="language_family")
    results = bor.train(thresholds=[0.5, 0.6, 0.7, 0.8], verbose=True)
    bor.output(
        'tsv',
        filename=our_path("store", "CL-train-CV10-fold-00-train"),
        prettify=False, ignore="all")

    wl = LexStat('splits/CV10-fold-00-train.tsv')
    bor.predict_on_wordlist(wl)
    wl.output(
        'tsv',
        filename=our_path("store", "CL-predict-CV10-fold-00-train"),
        prettify=False, ignore="all")

    wl = LexStat('splits/CV10-fold-00-test.tsv')
    bor.predict_on_wordlist(wl)
    wl.output(
        'tsv',
        filename=our_path("store", "CL-predict-CV10-fold-00-test"),
        prettify=False, ignore="all")

    print("* Training Results *")
    results += [[bor.best_idx, bor.best_threshold, bor.best_f1score]]
    print(tabulate(results, headers=["iter", "threshold", "F1 score"]))
