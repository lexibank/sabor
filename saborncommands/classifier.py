"""
Borrowings by Classifier
"""
from lingpy import *
from lexibank_sabor import (
        get_our_wordlist, sca_distance, edit_distance,
        evaluate_borrowings,
        our_path)
from sklearn.svm import SVC
from functools import partial
import collections

from tabulate import tabulate


def distance_by_idx(idxA, idxB, wordlist, distance=None, **kw):
    return distance(
            wordlist[idxA, "tokens"],
            wordlist[idxB, "tokens"],
            **kw)


clf_sca = partial(
        distance_by_idx,
        distance=sca_distance)

clf_sca_ov = partial(
        distance_by_idx,
        distance=sca_distance,
        mode='overlap')

clf_sca_lo = partial(
        distance_by_idx,
        distance=sca_distance,
        mode='local')

clf_ned = partial(
        distance_by_idx,
        distance=edit_distance)


# One-hot encoding, generalized to support value besides 1.
def one_hot(idx, sz, value=1):
    coded = [0] * sz
    coded[idx] = value
    return coded


# One hot vector for target language. Supports value besides 1.
def tar_one_hot(idx, wl, value=1):
    return one_hot(wl.cols.index(
        wl[idx, "doculect"]), len(wl.cols), value)


class ClassifierBasedBorrowingDetection(LexStat):

    def __init__( 
            self, 
            infile,
            donors,
            clf=None,
            funcs=None,
            props=None,
            props_tar=None,
            family="family",
            segments="tokens",
            ipa="form",
            known_donor="donor_language",
            **kw
            ):
        """

        """
        LexStat.__init__(self, infile, ipa=ipa, segments=segments, **kw)
        self.clf = clf or SVC(kernel="linear")
        # Default is linear kernel SVC. Other kernels are rbf, poly, sigmoid.
        self.funcs = funcs or [clf_sca, clf_ned]
        # Funcs involve both donor and target.

        # Prop args: x is the entry index, y is the wordlist reference.
        self.props = props if props is not None else [
            # Applied to both donor and target.
            lambda x, y: len(y[x, "tokens"])
            ]
        self.props_tar = props_tar if props_tar is not None else [
            # Applied only to target.
            tar_one_hot,
            ]
        # If multiple donors, could also use donor specific properties.
        # one_hot for donors could be indexed off of donors list.

        self.donors = [donors] if isinstance(donors, str) else donors
        # Define wordlist field names.
        self.family = family
        self.segments = segments
        self.known_donor = known_donor
        self.donor_families = {fam for (ID, lang, fam)
                               in self.iter_rows('doculect', self.family)
                               if lang in self.donors}

    @staticmethod
    def wrap_list(value):
        return value if isinstance(value, list) else [value]

    def get_x_row(self, idx_don, idx_tar, wl):
        row = []
        for prop in self.props:
            row += self.wrap_list(prop(idx_don, wl))
            row += self.wrap_list(prop(idx_tar, wl))
        for prop in self.props_tar:
            row += self.wrap_list(prop(idx_tar, wl))
        for func in self.funcs:
            row += [func(idx_don, idx_tar, wl)]
        return row

    def train(self, verbose=False, log=None):
        """
        Train the classifier.
        """
        # set up all pairs for the training procedure
        dt, pairs, result = {}, [], []
        # dt is separated donor, target lists of indices by concept.
        for concept in self.rows:
            dt[concept] = [[], []]
            idxs = self.get_list(row=concept, flat=True)
            dt[concept][0] = [
                    idx for idx in idxs if self[idx, self.family] in
                    self.donor_families]
            dt[concept][1] = [
                    idx for idx in idxs if idx not in dt[concept][0]]
            for idxA in dt[concept][0]:
                for idxB in dt[concept][1]:
                    # pairs is donor-target pairs of indices
                    # for the same concepts.
                    # idxA is for donor; idxB is for target.
                    pairs += [[idxA, idxB]]
                    if self[idxA, "doculect"] == self[idxB, self.known_donor]:
                        # result is whether source donor language
                        # is same as known donor language for pair.
                        result += [1]
                    else:
                        result += [0]

        # form the training matrix corresponding to donor-target pairs.
        matrix = []
        for idxA, idxB in pairs:
            matrix += [self.get_x_row(idxA, idxB, self)]

        if verbose:  # Look at first few rows.
            log.info("Length of x {}.".format(len(matrix[0])))
            log.info(matrix[:5])

        self.clf.fit(matrix, result)
        if verbose:
            log.info("Trained the classifier.")

    def predict(self, donors, targets, wordlist):
        hits = collections.defaultdict(list)
        for idxA in donors:
            for idxB in targets:
                pred = self.clf.predict(
                    [self.get_x_row(idxA, idxB, wordlist)])
                hits[idxB] += [(idxA, pred)]

        out = {}
        for hit, pairs in hits.items():
            out[hit] = sorted(pairs, key=lambda x: x[1][0], reverse=True)[0]
            out[hit] = (out[hit][0], out[hit][1][0])
        return out

    def predict_on_wordlist(self, wordlist):
        B = {idx: "" for idx in wordlist}
        for concept in wordlist.rows:
            idxs = wordlist.get_list(row=concept, flat=True)
            donors = [idx for idx in idxs if wordlist[idx, self.family] in
                      self.donor_families]
            targets = [idx for idx in idxs if idx not in donors]
            hits = self.predict(donors, targets, wordlist)
            for hit, pair in hits.items():
                B[hit] = pair[1]
        wordlist.add_entries('source_language', B, lambda x: x)


def run(args):
    # wl = get_our_wordlist()

    wl = Wordlist(our_path("splits", "CV10-fold-00-train.tsv"))

    wl_test = Wordlist(our_path("splits", "CV10-fold-00-test.tsv"))
    args.log.info("Loaded train and test wordlists.")

    def analyze_borrowing(wl, wl_test):
        bor = ClassifierBasedBorrowingDetection(
            wl, donors="Spanish",
            clf=SVC(kernel="linear"),
            funcs=[clf_sca, clf_ned],
            by_tar=False,
            family="language_family")
        bor.train(verbose=False, log=args.log)
        bor.predict_on_wordlist(wl_test)
        args.log.info("Evaluation:" + str(evaluate_borrowings(
            wl_test,
            "source_language",
            bor.known_donor,
            bor.donors,
            bor.donor_families,
            family=bor.family)))

        # concepts = bor.rows[:30]
        # table = []
        # for concept in concepts:
        #    if bor.dt[concept][0] and bor.dt[concept][1]:
        #        for a, b in bor.predict(bor.dt[concept][0], bor.dt[concept][1], bor).items():
        #            table += [[concept, bor[a, "doculect"], str(bor[a, "tokens"]), b[1], bor[a,
        #                bor.known_donor]]]
        # print(tabulate(table, tablefmt="plain"))

    print("* overall *")
    analyze_borrowing(wl, wl_test)
    # Store test results.
    file_path = 'store/test-new-predict-CV10-fold-00-test'
    wl_test.output("tsv", filename=file_path, prettify=False, ignore="all")
    #  Ooops.  Output of source_language is 0/1. No output of source_id.
