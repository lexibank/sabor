"""
Closest donor match based on pairwise sequence comparison.
Experiment without restriction to same concept.
"""
# import collections
from functools import partial
from collections import defaultdict
from tabulate import tabulate
import ast
from lexibank_sabor import (
        get_our_wordlist, our_path,
        sca_distance, edit_distance,
        # evaluate_borrowings_fs,
        evaluate_borrowings,
        subset_wl,
        get_language_list
)
from pylexibank import progressbar
from lingpy import *
import pyconcepticon.api


class CentralConceptLookup:
    """
    Construct lookup tables for concept to central concept, and
    for central concept to concept. Construct list of indices of
    entries for each central concept. Wordlist is assumed to be
    already censored to just donor entries. Lookup returns list of
    indices that correspond to central concept.
    """
    def __init__(
            self,
            wordlist,
            concepticon_ref=None,
            concept="concept"
            ):
        """
        :param wordlist: Wordlist censored to just donor entries.
        :param concepticon_ref: Reference to concepticon table of
        id, concept, central_concept
        :param concept: name of concept field in wordlist.

        Construct:
        - concept to central concept
        - central concept to concepts
        - central concept to indices
        Function:
        - lookup indices for concept via central concept.
          1. lookup central concept given concept.
          4. get and return indices
        """

        concepts = self.get_concept_table(concepticon_ref)

        self.concept = concept
        self.concept_central = dict()
        self.central_concepts = defaultdict(list)
        for _, concept, central in concepts:
            self.concept_central[concept] = central
            self.central_concepts[central].append(concept)
        self.central_indices = defaultdict(set)
        missing = 0
        for idx in wordlist:
            concept = wordlist[idx, self.concept]
            central = self.concept_central.get(concept)
            if not central:  # Concept not in database.
                print("Missing concept in central concepts; added", concept)
                self.central_concepts[concept].append(concept)
                self.concept_central[concept] = concept
                central = concept
                missing += 1
            self.central_indices[central].add(idx)

        if missing: print(f"Missing {missing:d} concepts; copied to central")

    @staticmethod
    def get_concept_table(concepticon_ref=None):
        concepticon = pyconcepticon.api.Concepticon(concepticon_ref)
        concept_list = concepticon.conceptlists["Rzymski-2020-1624"]
        concepts = list(concept_list.concepts.values())
        table = []
        for concept in concepts:
            table.append([concept.concepticon_id,
                          concept.concepticon_gloss,
                          concept.attributes['central_concept']])
        return table

    def lookup(self, concept):
        central = self.concept_central[concept]
        return list(self.central_indices[central])


def calculate_donor_match(
        target_tks,
        candidates,
        func=None):
    min_score = 1.0
    min_idx = 0
    # min_tks = ""
    for donor_idx, donor_tks in candidates.items():
        score = func(donor_tks, target_tks)
        if score < min_score:
            min_score = score
            min_idx = donor_idx
            # min_tks = donor_tks
    # print("target", target_tks, "candidate", min_idx, min_score, min_tks)
    return min_idx, min_score


def get_closest_donor_match_value_exp(
        wordlist,
        donors,
        concept="concept",
        segments="tokens",
        func=None,
        restrict_concept='target'
        ):
    """
    Calculate closest match in pairwise comparison of donor and target words.
    Censor all results > threshold as not useful match.
    Experiment pairwise without restriction to same concept.
    Could make multiple step process to screen on only likely matches
    if there were a cheap initial method to use before pairwise alignment.

    :param wordlist: LingPy wordlist.
    :param donors: Donor languages, passed as a list.
    :param concept: Column in which concept information is given in the wordlist.
    :param segments: Column in which segmented IPA tokens are given in the wordlist.
    :param func: Function comparing two sequences and returning a distance
      score (defaults to sca_distance).
    :param restrict_concept: Whether to restrict match to target concept.
    """
    func = func or sca_distance

    # get concept slots from the data (in case we use broader concepts by clics
    # communities), we essentially already split data in donor indices and
    # target indices by putting target indices in a list.

    # Experiment is to not restrict donor indices.  Let's try in a way that later
    # we might screen on how close donor concept is to target concept.

    concepts = defaultdict(list)

    target_candidates = defaultdict(set)
    central_candidates = defaultdict(set)
    all_candidates = {idx: wordlist[idx, segments] for idx in wordlist
                      if wordlist[idx, 'doculect'] in donors}

    central_concept = None
    if restrict_concept in ['central', 'nested']:
        wl_donors = subset_wl(wordlist, donors)
        central_concept = CentralConceptLookup(wl_donors)
    for idx in wordlist:
        if wordlist[idx, "doculect"] not in donors:
            concepts[wordlist[idx, concept]].append(idx)
        else:  # donor
            concept_value = wordlist[idx, concept]
            target_candidates[concept_value].add(idx)
            if restrict_concept in ['central', 'nested']:
                central_candidates[concept_value].update(
                    central_concept.lookup(concept_value))

    # calculate pairwise values for donors with each target.
    # keep only the closest match.
    # Only need target distance if target mode selected,
    # Only need central distance if central mode selected,
    # Need target and central distance if nested selected,

    items = list(concepts.items())
    concepts = {key: value for key, value in items}

    closest = dict()
    for target_concept, target_indices in progressbar(
            concepts.items(), desc="calculating closest match on borrowings"):
        if restrict_concept == 'none':
            candidates = all_candidates
        elif restrict_concept == 'central':
            candidates = {idx: all_candidates[idx]
                          for idx in central_candidates[target_concept]}
        else:  # 'target', 'nested'
            candidates = {idx: all_candidates[idx]
                          for idx in target_candidates[target_concept]}

        if restrict_concept != 'nested':
            for target_idx in target_indices:
                closest[target_idx] = calculate_donor_match(
                    wordlist[target_idx, segments], candidates, func)
        else:  # 'nested'
            nested_candidates = {idx: all_candidates[idx] for idx in
                                 central_candidates[target_concept] -
                                 target_candidates[target_concept]}
            for target_idx in target_indices:
                target = calculate_donor_match(
                    wordlist[target_idx, segments], candidates, func)
                nested = calculate_donor_match(
                    wordlist[target_idx, segments], nested_candidates, func)
                closest[target_idx] = (target, nested)

    return closest

    # Test print
    # for idx in wordlist:
    #     if wordlist[idx, "candidate"]:
    #         print(idx, wordlist[idx, 'doculect'],
    #               wordlist[idx, 'distance'],
    #               wordlist[idx, 'candidate'])
    #         print(wordlist[wordlist[idx, 'candidate']])


sca_gl = partial(
        sca_distance,
        mode='global')
sca_gl.__name__ = 'closest_match_sca_global'

sca_ov = partial(
        sca_distance,
        mode='overlap')
sca_ov.__name__ = 'closest_match_sca_overlap'

sca_lo = partial(
        sca_distance,
        mode='local')
sca_lo.__name__ = 'closest_match_sca_local'

ned = edit_distance


class ClosestDonorMatchExp(Wordlist):
    def __init__(
            self,
            infile,
            donors,
            func=None,
            segments="tokens",
            known_donor="donor_language",
            restrict_concept='target',
            **kw
            ):
        """
        Function allows to test thresholds to identify borrowings with the \
                simple_donor_search function.
        """
        Wordlist.__init__(self, infile, **kw)
        self.func = func or sca_gl
        self.donors = [donors] if isinstance(donors, str) else donors

        # Define wordlist field names.
        self.segments = segments
        self.known_donor = known_donor
        self.restrict_concept = restrict_concept

        self.best_value = 0
        self.best_score = 0
        self.best_key = 'threshold'

    def construct_wordlist(self, infile):
        """
        Make borrowing classifier wordlist for other file prediction.
        Required in order to reference LexStat columns and trained scorer.
        :param infile: wordlist or wordlist file reference.
        :return: ClosestDonorMatchExp object.
        """
        return ClosestDonorMatchExp(
            infile,
            self.donors,
            func=self.func,
            segments=self.segments,
            known_donor=self.known_donor,
            restrict_concept=self.restrict_concept
        )

    @staticmethod
    def process_distance_versus_threshold(
            wl,
            closest,
            threshold=None,
            donor_lng="source_language",
            donor_id="source_id"):

        wl.add_entries(
            'distance', "doculect", lambda x: None, override=True)
        wl.add_entries(
            donor_lng, "doculect", lambda x: "", override=True)
        wl.add_entries(
            donor_id, "doculect", lambda x: "", override=True)
        for idx in wl:
            item = closest.get(idx)
            if item is None: continue
            if type(item[0]) is int:  # target, central, or none
                candidate, distance = item
                if distance < threshold:
                    wl[idx, 'distance'] = distance
                    wl[idx, donor_id] = candidate
                    wl[idx, donor_lng] = wl[candidate, "doculect"]

            if type(item[0]) is tuple:
                # no target match, but possible central match.
                candidate, distance = item[0]
                if distance < threshold:
                    wl[idx, 'distance'] = distance
                    wl[idx, donor_id] = candidate
                    wl[idx, donor_lng] = wl[candidate, "doculect"]
                else:  # test central since no target match
                    candidate, distance = item[1]
                    if distance < threshold-0.25:  # penalty for central.
                        wl[idx, 'distance'] = distance
                        wl[idx, donor_id] = candidate
                        wl[idx, donor_lng] = wl[candidate, "doculect"]

        # Test print
        # for idx in wl:
        #     if wl[idx, 'source_id']:
        #         print(idx, wl[idx, 'doculect'],
        #               wl[idx, 'distance'],
        #               wl[idx, donor_id],
        #               wl[idx, donor_lng])

    def train(self, thresholds=None, verbose=False):
        """
        Train the threshold on the current data.
        """
        thresholds = thresholds or [round(i * 0.05, 3) for i in range(1, 20)]

        # calculate distances between all pairs, only once.
        # return dict of distance for each entry id.
        closest = get_closest_donor_match_value_exp(
            self,
            self.donors,
            func=self.func,
            restrict_concept=self.restrict_concept)
        if verbose: print("computed distances")

        best_t, best_f = 0, 0
        scores_table = []
        for i, threshold in enumerate(thresholds):
            self.process_distance_versus_threshold(self, closest, threshold)

            if self.known_donor in self.columns:
                # Calculate F score if known donor column.
                scores = evaluate_borrowings(
                    self,
                    "source_language",
                    self.known_donor,
                    self.donors)
                fs = scores['f1']
                if fs > best_f:
                    best_t = threshold
                    best_f = fs
                if verbose: print("Threshold:", threshold, "; Scores:", scores)
                scores['threshold'] = threshold
                scores_table.append(scores)

        self.best_value = best_t
        self.best_score = best_f
        if len(thresholds) > 1:  # Refresh best source language and id
            self.process_distance_versus_threshold(self, closest, best_t)

        return scores_table

    def predict_on_wordlist(
            self, wl,
            donor_lng="source_language",
            donor_id="source_id"):
        """
        Predict for an entire wordlist.
        """
        closest = get_closest_donor_match_value_exp(
            wl,
            self.donors,
            func=self.func,
            restrict_concept=self.restrict_concept)

        self.process_distance_versus_threshold(
            wl, closest, self.best_value, donor_lng, donor_id)


def register(parser):
    parser.add_argument(
        "--function",
        type=str,
        default="sca",
        choices=["sca", "ned", "sca_ov", "sca_lo"],
        help="select edit distance (ned), or sound correspondence "
             "alignment (sca), or sca with overlap (sca_ov), "
             "or local (sca_lo)."
        )
    parser.add_argument(
        "--threshold",
        default=[round(i*0.05, 3) for i in range(1, 20)],
        nargs="*",
        type=float,
        help="threshold distances to determine whether likely borrowing."
    )
    parser.add_argument(
        "--file",
        default=None,  # e.g., "splits/CV10-fold-00-train.tsv"
        help="wordlist filename containing donor and target language tokens."
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
        help="subset of languages to include; default is all languages."
    )
    parser.add_argument(
        "--donor",
        type=str,
        nargs="*",
        default=["Spanish"],
        help="Donor languages for focused analysis."
    )
    parser.add_argument(
        "--label",
        type=str,
        default='',
        help="Qualifier label for output."
    )
    parser.add_argument(
        "--restrict",
        type=str,
        default='target',
        choices=['target', 'central', 'nested', 'none'],
        help="Select donor matching scope restriction: "
             "target or central concept, none, nested target[central]."
    )


def run(args):

    function = {"ned": ned,
                "sca": sca_gl,
                "sca_ov": sca_ov,
                "sca_lo": sca_lo}[args.function]
    # TODO : if both train and test files specified, complete donor entries
    #  should be made available in both files.  This is to ensure that target
    #  languages are tested versus the complete vocabulary of donor entries.
    if args.file:
        wl = Wordlist(args.file)
        args.log.info("Construct closest from {fl}.".format(fl=args.file))
    else:
        wl = get_our_wordlist()
        args.log.info("Construct closest from SaBor database.")

    if args.language:
        args.language = get_language_list(args.language, args.donor)
        wl = subset_wl(wl, args.language)
        args.log.info("Subset of languages: {}".format(args.language))

    # restrict = ast.literal_eval(args.restrict)
    bor = ClosestDonorMatchExp(wl, func=function, donors=args.donor,
                               restrict_concept=args.restrict)

    results = bor.train(thresholds=args.threshold, verbose=False)
    args.log.info("Trained with donors {d}, function {func}".
                  format(d=bor.donors, func=bor.func.__name__))
    args.log.info("Best: threshold {thr:.2f}, F1 score {f1:.3f}".
                  format(thr=bor.best_value, f1=bor.best_score))

    full_name = "CM-EXP-sp-predict-{func}-{thr:.2f}-{label}-{r}-train".\
        format(func=bor.func.__name__, thr=bor.best_value,
               label=args.label, r=args.restrict)
    file_path = our_path("store", full_name)
    bor.output("tsv", filename=file_path, prettify=False, ignore="all")
    # No need to predict on wordlist, since required fields already written.

    if args.testfile:
        wl = Wordlist(args.testfile)
        args.log.info("Test closest from {fl}.".format(fl=args.testfile))

        if args.language:
            wl = subset_wl(wl, args.language)
            args.log.info("Subset of languages: {}".format(args.language))

        wl = bor.construct_wordlist(wl)
        bor.predict_on_wordlist(wl)

        # Just to remind us after so many cluster messages.
        args.log.info("Trained with donors {d}, function {func}".
                      format(d=bor.donors, func=bor.func.__name__))
        args.log.info("Best: threshold {thr:.2f}, F1 score {f1:.3f}".
                      format(thr=bor.best_value, f1=bor.best_score))
        scores = evaluate_borrowings(wl, "source_language",
                                     bor.known_donor, bor.donors)
        print("Scores:", scores)
        args.log.info("Test: threshold {thr:.2f}, F1 score {f1:.3f}".
                      format(thr=bor.best_value, f1=scores['f1']))

        full_name = "CM-sp-predict-{func}-{thr:.2f}-{label}-{r}-test".\
            format(func=bor.func.__name__, thr=bor.best_value,
                   label=args.label, r=args.restrict)
        file_path = our_path("store", full_name)
        wl.output("tsv", filename=file_path, prettify=False, ignore="all")

    # Exploring
    print()
    print("Borrowing detection results by threshold.")
    print(tabulate(results, headers="keys", tablefmt="pip", floatfmt=".3f"))
