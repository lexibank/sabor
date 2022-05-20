"""
Borrowings by Classifier
"""
from lingpy import *
from lexibank_sabor import Dataset as SABOR
from lexibank_sabor import (
        get_our_wordlist, sca_distance, edit_distance,
        simple_donor_search,
        evaluate_borrowings_fs, 
        our_path,
        sds_by_concept)
from sklearn.svm import SVC
from functools import partial
import collections

from tabulate import tabulate


def distance_by_idx(idxA, idxB, wordlist, distance=None, **kw):
    return distance(
            wordlist[idxA, "tokens"],  # wordlist.segments],
            wordlist[idxB, "tokens"],  # wordlist.segments],
            **kw)


clf_sca = partial(
        distance_by_idx,
        distance=sca_distance)
clf_ned = partial(
        distance_by_idx,
        distance=edit_distance)
        

class ClassifierBasedBorrowingDetection(LexStat):

    def __init__( 
            self, 
            infile,
            donors,
            clf=None,
            funcs=None,
            props=None,
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
        self.funcs = funcs or [clf_sca, clf_ned]

        self.props = props or [
                lambda x, y: y.cols.index(y[x, "doculect"]),
                lambda x, y: len(y[x, "tokens"])
                ]
        self.donors = [donors] if isinstance(donors, str) else donors
        # Define wordlist field names.
        self.family = family
        self.segments = segments
        self.known_donor = known_donor
        self.donor_families = {fam for (ID, lang, fam)
                               in self.iter_rows('doculect', self.family)
                               if lang in self.donors}

    def train(self, verbose=False):
        """
        Train the classifier.
        """
        # set up all pairs for the training procedure
        self.dt, self.pairs, self.result = {}, [], []
        for concept in self.rows:
            self.dt[concept] = [[], []]
            idxs = self.get_list(row=concept, flat=True)
            self.dt[concept][0] = [
                    idx for idx in idxs if self[idx, self.family] in
                    self.donor_families]
            self.dt[concept][1] = [
                    idx for idx in idxs if idx not in self.dt[concept][0]]
            for idxA in self.dt[concept][0]:
                for idxB in self.dt[concept][1]:
                    self.pairs += [[idxA, idxB]]
                    if self[idxA, "doculect"] == self[idxB, self.known_donor]:
                        self.result += [1]
                    else:
                        self.result += [0]
        matrix = []
        for idxA, idxB in self.pairs:
            row = []
            for prop in self.props:
                row += [prop(idxA, self)]
                row += [prop(idxB, self)]
            for func in self.funcs:
                row += [func(idxA, idxB, self)]
            matrix += [row]
        self.matrix = matrix
        self.clf.fit(self.matrix, self.result)
        if verbose: print("fitted the classifier")


    def predict(self, donors, targets, wordlist):
        hits = collections.defaultdict(list)
        for idxA in donors:
            for idxB in targets:
                row = []
                for prop in self.props:
                    row += [prop(idxA, wordlist)]
                    row += [prop(idxB, wordlist)]
                for func in self.funcs:
                    row += [func(idxA, idxB, wordlist)]
                pred = self.clf.predict([row])
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
    #wl = get_our_wordlist()
    wl = Wordlist(our_path("splits", "CV10-fold-00-train.tsv"))
    wl2 = ClassifierBasedBorrowingDetection(our_path("splits",
        "CV10-fold-00-test.tsv"),
        donors="Spanish", family="language_family"
        )

    args.log.info("loaded wordlist")
    bor = ClassifierBasedBorrowingDetection(
            wl, donors="Spanish", funcs=[clf_sca], family="language_family",
            clf=SVC(kernel="linear")
            )
    bor.train(verbose=True)
    bor.predict_on_wordlist(wl2)
    print(evaluate_borrowings_fs(
        wl2,
        "source_language",
        bor.known_donor, 
        bor.donors, 
        bor.donor_families,
        family=bor.family))
        #concepts = bor.rows[:30]
        #table = []
        #for concept in concepts:
        #    if bor.dt[concept][0] and bor.dt[concept][1]:
        #        for a, b in bor.predict(bor.dt[concept][0], bor.dt[concept][1], bor).items():
        #            table += [[concept, bor[a, "doculect"], str(bor[a, "tokens"]), b[1], bor[a,
        #                bor.known_donor]]]
        #print(tabulate(table, tablefmt="plain"))
        

