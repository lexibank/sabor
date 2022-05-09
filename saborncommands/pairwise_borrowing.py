"""
Pairwise alignment method approach to borrowing detection between
donor (intruder) language and multiple receiver (target) languages.

Using LingPy wordlists.

May 6, 2022
"""

import tempfile
from pathlib import Path

from lexibank_sabor import Dataset as SABOR

import collections
from lingpy import *
from pylexibank import progressbar as pb


class PairwiseBorrowing(Wordlist):
    def __init__(self, infile, func, donor, **kw):
        Wordlist.__init__(self, infile, **kw)
        self.func = func
        self.donor = donor
        self.current_thr = None
        self.best_thr = None
        self.best_score = None

        # Define wordlist field names.
        self.family = "language_family"
        self.concept = "concept"
        self.segments = "tokens"
        self.gold_donor_language = "donor_language"
        self.gold_donor_value = "donor_value"
        self.source_id_ = "source_id_"
        self.source_score_ = "source_score_"
        self.source_id = "source_id"

        def get_donor_family():
            # Can optimize function to query wordlist sparsely.
            return list({fam for (ID, lang, fam)
                        in self.iter_rows('doculect', self.family)
                        if lang.startswith(self.donor)})[0]

        self.donor_family = get_donor_family()

    @staticmethod
    def from_wordlist(wordlist, func, donor, **kw):
        # Write to flat file, create PairwiseBorrowing from flat file.
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp).joinpath('temp_wl').as_posix()
            wordlist.output('tsv', filename=file_path,
                            ignore='all', prettify=False)
            file_path += '.tsv'
            return PairwiseBorrowing(infile=file_path, func=func,
                                     donor=donor, **kw)

    def get_borrowings(self, threshold=0.3, ref="bor_id"):
        # Calculate the scores if necessary.
        if (self.source_score_ not in self.columns) or \
                (self.source_id_ not in self.columns):
            self.calc_distances()
        # Add entries for 'ref' when score less than threshold
        self.add_entries(ref, self.source_id_,
                         lambda x: x if x != 0 else "", override=True)
        for (ID, source, score) in self.iter_rows(
                self.source_id_, self.source_score_):
            self[ID, ref] = source \
                if type(score) is float and score < threshold else ""
        self.current_thr = threshold

    def calc_distances(self):
        concepts = {concept: [[], []] for concept in set(
            [self[idx, self.concept] for idx in self])}
        for idx in self:
            if self[idx, "doculect"].startswith(self.donor):
                concepts[self[idx, self.concept]][0] += [idx]
            # languages from donor family are not target languages.
            elif self[idx, self.family] != self.donor_family:
                concepts[self[idx, self.concept]][1] += [idx]

        # iterate over concepts and identify potential borrowings
        B = {idx: 0 for idx in self}
        D = {idx: None for idx in self}
        for concept, (donor_indices, target_indices) in pb(concepts.items()):
            # hits is a dictionary with target ID as key and list of possible donor
            # candidate ids as value
            hits = collections.defaultdict(list)
            for donor_idx in donor_indices:
                for target_idx in target_indices:

                    score = self.func(self[donor_idx, self.segments],
                                      self[target_idx, self.segments])
                    hits[target_idx] += [(donor_idx, score)]
            # get the minimum score hit, as we can have only one donor
            for hit, pairs in hits.items():
                B[hit], D[hit] = sorted(pairs, key=lambda x: x[1])[0]

        self.add_entries(self.source_id_, B,
                         lambda x: x if x != 0 else "")
        self.add_entries(self.source_score_, D,
                         lambda x: x if x is not None else "")

    def train(self, thresholds=None):
        if thresholds is None:
            thresholds = [i*0.05 for i in range(20)]
        best_thr, best_score = 0.0, 0.0
        pred_donor_ref = self.source_id
        for t in thresholds:
            self.get_borrowings(threshold=t, ref=pred_donor_ref)
            # calculate internal evaluation metric(s).
            score = self.evaluate_borrowings(
                pred_donor_ref, self.gold_donor_language)
            if score > best_score: best_thr, best_score = t, score
        self.best_thr = best_thr
        self.best_score = best_score
        # Set current_thr and source_id to optimal value.
        self.get_borrowings(threshold=self.best_thr, ref=self.source_id)

    def evaluate_borrowings(self, pred_ref, gold_ref):
        # Defensive programming:
        # Check for None and be sure of pred versus gold.
        # Return F1 score overall.
        fn = fp = tn = tp = 0
        for ID, pred_idx, gold_lang in self.iter_rows(pred_ref, gold_ref):
            if not pred_idx:
                if not gold_lang: tn += 1
                elif gold_lang.startswith(self.donor): fn += 1
                else: tn += 1
            elif pred_idx:
                if not gold_lang: fp += 1
                elif gold_lang.startswith(self.donor): tp += 1
                else: fp += 1

        return tp/(tp + (fp + fn)/2)

    def predict(self, words, languages):
        ...
        # Assumes a configured and trained model. Specific function provided
        # at init, optimized for threshold based on training data in wordlist.
        # Predicts borrowing status (?) based on comparison of input entries
        # for receiver languages aligned with donor language over concepts.
        # Output is predictions of borrowing from donor language.
        # Indicated by corresponding donor entry pointed to by 'source_id'.
        # For convenience 'source_language' and source_segments' are also shown.


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


def run_analysis(pairwise, name, threshold, log, report=True):
    """
    Shortcut for configuring and running an analysis.
    """

    full_name = "{name}-{func}-{thr:.2f}".format(
        name=name, func=pairwise.func.__name__, thr=threshold)
    if report:
        log.info("# ANALYSIS: T={0:.2f}, D={1}, F={2}".format(
            threshold, pairwise.func.__name__, full_name))
        log.info("## running experiment {}".format(full_name))

    pairwise.get_borrowings(threshold=threshold, ref="source_id")

    if report:
        file_path = str(SABOR().dir / "store" / full_name)
        pairwise.output("tsv", filename=file_path, prettify=False, ignore="all")
        log.info("## found {0} borrowings".format(
            len([x for x in pairwise if pairwise[x, "source_id"]])))
        log.info("---")


def get_sabor_wordlist():
    wl = Wordlist.from_cldf(
        str(SABOR().cldf_dir / "cldf-metadata.json"),
        columns=[
            "language_id", "language_family",
            "concept_name", "value", "form", "segments",
            "donor_language", "donor_value"],
    )
    # donor_language and donor_value fields read as None when empty.
    for idx in wl:
        if wl[idx, "donor_language"] is None: wl[idx, "donor_language"] = ""
        if wl[idx, "donor_value"] is None: wl[idx, "donor_value"] = ""
    return wl


def register(parser):
    parser.add_argument(
        "--infile",
        default=None,  # "splits/CV10-fold-00-train.tsv",
        help="wordlist filename containing donor and target language tokens"
    )
    parser.add_argument(
        "--function",
        default="SCA",
        choices=["SCA", "NED"],
        help="select Needleman edit distance, or sound correspondence alignment"
        )
    parser.add_argument(
        "--threshold",
        default=0.3,
        type=float,
        help="enter <= threshold distance to determine whether likely borrowing"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train pairwise to optimize F1 score"
    )


def run(args):
    function = {"NED": edit_distance,
                "SCA": sca_distance}[args.function]
    donor = "Spanish"

    if args.infile:
        pw = PairwiseBorrowing(
            infile=args.infile, func=function, donor=donor)
        args.log.info("Constructed pairwise from {fl}.".format(fl=args.infile))
    else:
        wl = get_sabor_wordlist()
        pw = PairwiseBorrowing.from_wordlist(
            wordlist=wl, func=function, donor=donor)

    if args.train:
        pw.train()
        args.log.info("Trained with donor {d}, function {func}".
                      format(d=pw.donor, func=pw.func))
        args.log.info("Best: threshold {thr:.2f}, F1 score {f1:.3f}".
                      format(thr=pw.best_thr, f1=pw.best_score))
    else:
        run_analysis(pw, name="pw-spa", threshold=args.threshold, log=args.log)
