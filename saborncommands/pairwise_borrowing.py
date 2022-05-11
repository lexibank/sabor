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
from lexibank_sabor import get_our_wordlist

import collections
from lingpy import *
from pylexibank import progressbar as pb


# static helper for is float convertible
def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# ============================================
# Definition of command and related functions.
# ============================================
def sca_distance(seqA, seqB, **kw):
    """
    Shortcut for computing SCA distances from two strings.
    """

    pair = Pairwise(seqA, seqB)
    pair.align(distance=True, **kw)

    return pair.alignments[0][-1]


def edit_distance(seqA, seqB, **kw):
    """
    Shortcut normalized edit distance.
    """
    return edit_dist(seqA, seqB, normalized=True)


def pairwise_comparison(
        wordlist,
        donors,
        family="language_family",
        concept="concept",
        segments="tokens",
        donor_lng="source_language",
        donor_id="source_id",
        func=None,
        threshold=0.45,
        **kw
        ):
    """
    Find borrowings by carrying out a pairwise comparison of donor and target words.

    :param wordlist: LingPy wordlist.
    :param donors: Donor languages, passed as a list.
    :param family: Column in which language family information is given in the wordlist.
    :param concept: Column in which concept information is given in the wordlist.
    :param segments: Column in which segmented IPA tokens are given in the wordlist.
    :param donor_lng: Column to which information on predicted donor languages will
      be written (defaults to "source_language").
    :param donor_id: Column to which we write information on the ID of the predicted donor.
    :param func: Function comparing two sequences and returning a distance
      score (defaults to sca_distance).
    :param threshold: Threshold, at which we recognize a word as being borrowed.
    """
    func = func or sca_distance

    # get concept slots from the data (in case we use broader concepts by clics
    # communities), we essentially already split data in donor indices and
    # target indices by putting them in a list.
    donor_families = {fam for (ID, lang, fam)
                      in wordlist.iter_rows('doculect', family)
                      if lang in donors}

    concepts = {concept: [[], []] for concept in set(
        [wordlist[idx, concept] for idx in wordlist])}
    for idx in wordlist:
        if wordlist[idx, "doculect"] in donors:
            concepts[wordlist[idx, concept]][0] += [idx]
        # languages from donor families are not target languages.
        elif wordlist[idx, family] not in donor_families:
            concepts[wordlist[idx, concept]][1] += [idx]

    # iterate over concepts and identify potential borrowings
    B = {idx: 0 for idx in wordlist}
    for concept, (donor_indices, target_indices) in pb(concepts.items(), 
            desc="searching for borrowings"):
        # hits is a dictionary with target ID as key and list of possible donor
        # candidate ids as value 
        hits = collections.defaultdict(list)
        for idxA in donor_indices:
            for idxB in target_indices:

                score = func(wordlist[idxA, segments],
                             wordlist[idxB, segments], **kw)
                if score < threshold:
                    hits[idxB] += [(idxA, score)]
        # we sort the hits, as we can have only one donor
        for hit, pairs in hits.items():
            B[hit] = sorted(pairs, key=lambda x: x[1])[0][0]

    wordlist.add_entries(
            donor_lng, B, lambda x: wordlist[x, "doculect"] if x != 0 else "")
    wordlist.add_entries(
            donor_id, B, lambda x: x if x != 0 else "")



class PairwiseBorrowing(Wordlist):
    def __init__(
            self, 
            infile,
            donor,
            func=None,
            family="family",
            segments="tokens",
            known_donor="donor_language",
            **kw
            ):
        Wordlist.__init__(self, infile, **kw)
        if not func:
            self.func = lambda x, y: sca_distance(x, y)
        else:
            self.func = func
            
        self.donor = donor

        # Define wordlist field names.
        self.family = family
        self.segments = segments
        self.known_donor = known_donor
        self.donor_family = [fam for (ID, lang, fam)
                        in self.iter_rows('doculect', self.family)
                        if lang == self.donor][0]
    
    def train(self, thresholds=None):
        """
        Train the threshold on the current data.
        """
        thresholds = thresholds or [i*0.1 for i in range(1, 10)]
        best_t, best_e = 0, 0
        for i, threshold in enumerate(thresholds):
            print("analyzing {0:.2f}".format(threshold))
            tidx = "t_"+str(i+1)
            pairwise_comparison(
                    self,
                    [self.donor],
                    family=self.family,
                    #concept=self.columns[self._rowidx],
                    segments="tokens",
                    donor_lng=tidx+"_lng",
                    donor_id=tidx+"_id",
                    func=self.func,
                    threshold=threshold
                    )
            fs = self.evaluate_borrowings(tidx+"_lng", self.known_donor)
            if fs > best_e:
                best_t = threshold
                best_e = fs
            print("... {0:.2f}".format(fs))
        self.best_t = best_t

    def evaluate_borrowings(self, pred, gold):
        """
        Return F-Scores for the donor detection.
        """
        # Defensive programming:
        # Check for None and be sure of pred versus gold.
        # Return F1 score overall.
        # Evaluation wordlist is from parent.
        fn = fp = tn = tp = 0
        for idx, pred_lng, gold_lng in self.iter_rows(pred, gold):
            if self[idx, self.family] != self.donor_family:
                if not pred_lng:
                    if not gold_lng: tn += 1
                    elif gold_lng == self.donor: fn += 1
                    else: tn += 1
                elif pred_lng:
                    if not gold_lng: fp += 1
                    elif gold_lng == self.donor: tp += 1
                    else: fp += 1

        return tp/(tp + (fp + fn)/2)



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
    bor = PairwiseBorrowing(
            wl, donor="Spanish", func=sca_distance, family="language_family")
    bor.train()
    print("best threshold is {0:.2f}".format(bor.best_t))
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

