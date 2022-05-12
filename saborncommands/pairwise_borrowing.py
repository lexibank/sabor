"""
Pairwise alignment method approach to borrowing detection between
donor (intruder) language and multiple receiver (target) languages.

Using LingPy wordlists.

May 6, 2022
"""

import tempfile
from pathlib import Path
import copy
import csv

from lexibank_sabor import Dataset as SABOR
from lexibank_sabor import (
        get_our_wordlist, sca_distance, edit_distance,
        simple_donor_search,
        evaluate_borrowings_fs, 
        sds_by_concept)

import collections
from lingpy import *
from pylexibank import progressbar as pb



class SimpleDonorSearch(Wordlist):
    def __init__(
            self, 
            infile,
            donors,
            func=None,
            family="family",
            segments="tokens",
            known_donor="donor_language",
            **kw
            ):
        """
        Function allows to test thresholds to identify borrowings with the \
                simple_donor_search function.
        """
        Wordlist.__init__(self, infile, **kw)
        if not func:
            self.func = lambda x, y: sca_distance(x, y)
        else:
            self.func = func
        self.donors = [donors] if isinstance(donors, str) else donors

        # Define wordlist field names.
        self.family = family
        self.segments = segments
        self.known_donor = known_donor
        self.donor_families = {fam for (ID, lang, fam)
                        in self.iter_rows('doculect', self.family)
                        if lang in self.donors}
    

    def train(self, thresholds=None, verbose=False):
        """
        Train the threshold on the current data.
        """
        thresholds = thresholds or [i*0.1 for i in range(1, 10)]
        
        # calculate distances between all pairs, only once, afterwards make a
        # function out of it, so we can pass this to the simple_donor_search
        # function
        D = {}
        for concept in self.concepts:
            idxs = self.get_list(row=concept, flat=True)
            donors = [idx for idx in idxs if self[idx, self.family] in \
                    self.donor_families]
            recips = [idx for idx in idxs if idx not in donors]
            for idxA in donors:
                for idxB in recips:
                    tksA, tksB = self[idxA, self.segments], self[
                            idxB, self.segments]
                    D[str(tksA), str(tksB)] = self.func(tksA, tksB)
        new_func = lambda x, y: D[str(x), str(y)]
        if verbose: print("computed distances")

        best_t, best_e = 0, 0
        for i, threshold in enumerate(thresholds):
            if verbose: print("analyzing {0:.2f}".format(threshold))
            tidx = "t_"+str(i+1)
            simple_donor_search(
                    self,
                    self.donors,
                    family=self.family,
                    concept=self._row_name,
                    segments="tokens",
                    donor_lng=tidx+"_lng",
                    donor_id=tidx+"_id",
                    func=new_func,
                    threshold=threshold
                    )
            fs = evaluate_borrowings_fs(
                    self, 
                    tidx+"_lng", 
                    self.known_donor,
                    self.donors,
                    self.donor_families,
                    family=self.family
                    )
            if fs > best_e:
                best_t = threshold
                best_e = fs
            if verbose: print("... {0:.2f}".format(fs))
        self.best_t = best_t

    def predict(self, donors, targets):
        """
        Predict borrowings for one concept.
        """
        return sds_by_concept(donors, targets, self.func, self.best_t)

    def predict_on_wordlist(
            self, wordlist, donor_lng="source_language", 
            donor_id="source_id"):
        """
        Predict for an entire wordlist.
        """
        simple_donor_search(
                wordlist,
                self.donors,
                family=self.family,
                concept=self._row_name,
                segments=self.segments,
                donor_lng=donor_lng,
                donor_id=donor_id,
                func=self.func,
                threshold=self.best_t)
        

def run_analysis(pairwise, name, threshold, log, report=True):
    """
    Shortcut for configuring and running an analysis with specific threshold.
    """
    full_name = "{name}-{func}-{thr:.2f}".format(
        name=name, func=pairwise.func.__name__, thr=threshold)
    if report:
        log.info("# ANALYSIS: T={0:.2f}, D={1}, F={2}".format(
            threshold, pairwise.func.__name__, full_name))

    pairwise.get_borrowings(threshold=threshold, ref="source_id",
                            ref_lang="source_language",
                            ref_segs="source_tokens")

    if report:
        file_path = str(SABOR().dir / "store" / full_name)
        cols = [col for col in pairwise.columns
                if col not in ['source_id_', 'source_score_']]
        pairwise.output("tsv", filename=file_path, prettify=False, ignore="all",
                        subset=True, cols=cols)
        log.info("## found {0} borrowings".format(
            len([x for x in pairwise if pairwise[x, "source_id"]])))
        log.info("---")


def register(parser):
    parser.add_argument(
        "--infile",
        default=None,  # e.g., "splits/CV10-fold-00-train.tsv"
        help="wordlist filename containing donor and target language tokens."
    )
    parser.add_argument(
        "--function",
        default="SCA",
        choices=["SCA", "NED"],
        help="select Needleman edit distance, or sound correspondence alignment."
        )
    parser.add_argument(
        "--threshold",
        default=0.3,
        type=float,
        help="enter <= threshold distance to determine whether likely borrowing."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train pairwise to optimize F1 score; gold language field required."
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="predict borrowings based on trained configuration."
    )
    parser.add_argument(
        "--testfile",
        default=None,  # e.g., "splits/CV10-fold-00-test.tsv"
        help="wordlist filename containing donor and target language tokens for test."
    )
    parser.add_argument(
        "--language",
        nargs="*",
        type=str,
        default=None,
        help="languages to use with predict; default is all languages."
    )


def run(args):
    wl = get_our_wordlist()
    args.log.info("loaded wordlist")
    bor = SimpleDonorSearch(
            wl, donors="Spanish", func=sca_distance, family="language_family")
    bor.train(verbose=True, thresholds=[i*0.05 for i in range(1,20)])
    print("best threshold is {0:.2f}".format(bor.best_t))
    hits = bor.predict(
            {"Spanish": ["m", "a", "n", "o"]}, 
            {
                "FakeX": ["m", "a", "n", "u", "ʃ", "k", "a"],
                "FakeY": ["p", "e", "p", "e", "l"]
                }
            )
    for idx, donor in hits.items():
        print(idx, donor)
    #function = {"NED": edit_distance,
    #            "SCA": sca_distance}[args.function]
    #donor = "Spanish"

    #if args.infile:
    #    pw = PairwiseBorrowing(
    #        infile=args.infile, func=function, donor=donor)
    #    args.log.info("Constructed pairwise from {fl}.".format(fl=args.infile))
    #else:
    #    wl = get_sabor_wordlist()
    #    pw = PairwiseBorrowing.from_wordlist(
    #        wordlist=wl, func=function, donor=donor)
    #    args.log.info("Constructed pairwise from SaBor cldf database.")

    #if args.train:
    #    pw.train()
    #    args.log.info("Trained with donor {d}, function {func}".
    #                  format(d=pw.donor, func=pw.func.__name__))
    #    args.log.info("Best: threshold {thr:.2f}, F1 score {f1:.3f}".
    #                  format(thr=pw.best_thr, f1=pw.best_score))
    #else:
    #    run_analysis(pw, name="pw-spa", threshold=args.threshold, log=args.log)

    #if args.predict:
    #    languages = "ALL" if args.language is None else args.language
    #    args.log.info("Predict with languages {l} and threshold {t}.".
    #                  format(l=languages, t=pw.best_thr))
    #    wl = pw.predict(testfile=args.testfile, languages=args.language)
    #    full_name = "pw-sp-predict-{func}-{thr:.2f}".format(
    #        func=pw.func.__name__, thr=pw.best_thr)
    #    file_path = str(SABOR().dir / "store" / full_name)
    #    cols = [col for col in wl.columns
    #            if col not in ['source_id_', 'source_score_']]
    #    wl.output("tsv", filename=file_path, prettify=False, ignore="all",
    #              subset=True, cols=cols)

