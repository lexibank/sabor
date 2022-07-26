"""
Markov language model calculates cross-entropy for words.
Smallest cross-entropy labels word for that model.
"""
from collections import defaultdict
import statistics

import attr
from nltk.util import ngrams, flatten
import nltk.lm as lm
import nltk.lm.preprocessing as pp

from lexibank_sabor import (
        get_our_wordlist, our_path,
        evaluate_borrowings_fs,
        # evaluate_borrowings,
        subset_wl,
        get_language_list
)
from lingpy import *
from pyclts import CLTS


def encode_segments_as_sca(wl, segments="tokens"):
    # Encode segments as soundclass('sca')
    clts = CLTS()
    for idx in wl:
        wl[idx, segments] = clts.soundclass("sca")(wl[idx, segments])


@attr.s
class MarkovWordModel:
    language = attr.ib(default=None)
    vocab = attr.ib(default=None)
    tokens = attr.ib(default=[], repr=False)
    model = attr.ib(default='kni')
    order = attr.ib(default=3)
    smoothing = attr.ib(default=0.1)
    direction = attr.ib(default='forward')
    """
    Use 2nd or 3rd order ngrams (1st or 2nd order dependency) to calculate word entropies.

    Notes
    -----
    Markov model is trained from list of tokens at time of construction.
    Entropy or entropy lists are calculated from the trained Markov model.

    Parameters
    ----------
    language : str, language name
    vocab : Vocabulary
    tokens : [[str]]
        List of language tokens in format:
            token as sequence of IPA character segments,
    model : str, optional
        Markov word model option from nltk. The default is "kni".
        kni = Kneser Ney interpolated,
        wbi = Witten Bell interpolated
        lp = Laplace,
        ls = Lidstone,
        ml = Maximum likelihood without smoothing.
    order : int, optional
        ngram order of model which counts symbol emitted and number
        of conditioning symbols. The default is 3.
    smoothing : float, optional
        Smoothing quantity for use by 'kni' or 'ls'. The default is 0.5.
    direction : str, optional
        Whether forward, backward.  Default is forward.

    """

    def __attrs_post_init__(self):
        if self.direction == 'backward':
            self.tokens = [token[::-1] for token in self.tokens]

        # NLTK recommended padding explicitly for training.
        train, vocab_ = pp.padded_everygram_pipeline(self.order, self.tokens)
        # Define the vocabulary if necessary,  allow for out of vocabulary.
        self.vocab = self.vocab or lm.Vocabulary(vocab_, unk_cutoff=2)

        # Define and then train the language model.
        options = {
            "kni": (lm.KneserNeyInterpolated, {"discount": self.smoothing}),
            "wbi": (lm.WittenBellInterpolated, {}),
            "lp": (lm.Laplace, {}),
            "ls": (lm.Lidstone, {"gamma": self.smoothing}),
            "mle": (lm.MLE, {}),
        }

        self.lm = options[self.model][0](
            order=self.order, vocabulary=self.vocab, **options[self.model][1]
        )
        self.lm.fit(train)

    def calculate_cross_entropies(self, tokens):
        """
        Calculate cross-entropies for list of tokens.
        :param tokens: list of tokens as space delimited IPA segments.
        :return: cross_entropies
        """
        return [self.calculate_cross_entropy(token) for token in tokens]

    def calculate_cross_entropy(self, token):
        """
        Calculate per sound average entropy of single token.
        """
        # return self.calculate_cross_entropies([token])[0]
        def calculate_cross_entropy_(token_, lm_):
            padded_token = list(pp.pad_both_ends(token_, n=self.order))
            ngrams_ = list(ngrams(padded_token, self.order))
            # Correction to exclude final transition with prob=1.
            # Cross-entropy is calculated base 2.
            return lm_.entropy(ngrams_)*len(ngrams_)/(len(ngrams_)-1)

        if self.direction == 'backward': token = token[::-1]
        return calculate_cross_entropy_(token, self.lm)


class CrossEntropyModel(Wordlist):
    # For now - shell to invoke Markov language model.
    def __init__(
            self,
            infile,
            donors,
            direction="forward",
            approach="dominant",
            segments="tokens",
            known_donor="donor_language",
            donor_lng="source_language",
            donor_id="source_id"
            ):
        """
            Invoke Markov chain to learn language models and calculate
            cross-entropies by language. Train to learn models and
            calculate cross-entropies or competing cross-entropies
            depending on approach.
            [all, dominant, borrowed, inherited, each]
        """
        Wordlist.__init__(self, infile)
        self.donors = [donors] if isinstance(donors, str) else donors
        self.direction = direction
        self.approach = approach
        # Define wordlist field names.
        self.segments = segments
        self.known_donor = known_donor
        self.donor_lng = donor_lng
        self.donor_id = donor_id

        # List recipient languages.
        self.languages = [lang for lang in self.cols
                          if lang not in self.donors]
        # define categories for cross-entropies.
        categories = [""]
        if self.approach == "borrowed":
            categories.append("borrowed")
        elif self.approach == "dominant":
            for donor in self.donors:
                categories.append(donor)
        self.categories = categories

        self.language_tokens = defaultdict(list)
        self.language_models = dict()

    def make_vocab(self):
        # Compile list of tokens and construct vocabulary by language.
        language_tokens_ = defaultdict(list)
        for lang, source in self.language_tokens.keys():
            language_tokens_[lang] += self.language_tokens[(lang, source)]

        language_vocabs = dict()
        for lang in language_tokens_:
            segments = flatten(language_tokens_[lang])
            language_vocabs[lang] = lm.vocabulary.Vocabulary(segments, unk_cutoff=2)
            # print(lang, language_vocabs[lang], sorted(language_vocabs[lang]))
        return language_vocabs

    def calculate_word_cross_entropies(self, idx, wl):
        lang = wl[idx, "doculect"]
        if lang not in self.languages: return None
        result = []
        for category in self.categories:
            lm_ = self.language_models[(lang, category)]
            if lm_:
                # Assume that only 1 donor language model per target language
                # Alternative might be to use some arbitrary high result value.
                value = lm_.calculate_cross_entropy(wl[idx, self.segments])
                result.append(value if value < 8.0 else 8.0)
        return result

    def train_markov_word_model(self, verbose=False):
        # Process by language for initial training of Markov chains.
        self.language_tokens = defaultdict(list)
        for idx in self:
            lang = self[idx, "doculect"]
            if lang not in self.languages: continue

            if self.approach == "all":
                source = ''
            else:
                source = self[idx, self.known_donor]
                if self.approach == "dominant":
                    source = source if source in self.donors else ''
                else:  # "borrowed"
                    source = "borrowed" if source else ''
            self.language_tokens[(lang, source)].\
                append(self[idx, self.segments])

        # Use a common vocabulary for training.
        language_vocabs = self.make_vocab()
        self.language_models = dict()
        for lang, source in self.language_tokens.keys():
            vocab = language_vocabs[lang]

            tokens = self.language_tokens.get((lang, source))
            self.language_models[(lang, source)] = \
                MarkovWordModel(lang, vocab, tokens,
                                direction=self.direction,
                                model='kni', smoothing=0.9)

            if verbose:
                # Calculated by the language model.  Not here.
                cross_entropies = self.language_models[(lang, source)]. \
                    calculate_cross_entropies(tokens)
                # counts = self.language_models[(lang, source)].lm.counts
                print("cross-entropies for {lang},{src}: "
                      "tokens {len}, avg {avg}, stdev {stdev}".
                      format(lang=lang, src=source,
                             len=len(tokens),
                             avg=statistics.mean(cross_entropies),
                             stdev=statistics.stdev(cross_entropies)))
        if verbose: print("trained Markov chains")


class LeastCrossEntropy(CrossEntropyModel):
    # For now - shell to invoke Markov language model.
    def __init__(
            self,
            infile,
            donors,
            direction="forward",
            approach="dominant",
            segments="tokens",
            known_donor="donor_language",
            donor_lng="source_language",
            donor_id="source_id"
            ):
        """
            Invoke Markov chain to learn language models and calculate
            cross-entropies by language. Train to learn models and
            calculate cross-entropies or competing cross-entropies
            depending on approach.
            [all, dominant, borrowed, inherited, each]
        """
        CrossEntropyModel.__init__(self, infile, donors, direction, approach,
                                   segments, known_donor, donor_lng, donor_id)

        self.best_value = None
        self.best_score = None

    def construct_wordlist(self, infile):
        """
        Make borrowing least_cross_entropy wordlist for other file prediction.
        :param infile: wordlist or wordlist file reference.
        :return: LeastCrossEntropy object.
        """
        return LeastCrossEntropy(
            infile,
            self.donors,
            segments=self.segments,
            known_donor=self.known_donor
        )

    def assign_least_donor(self, wl, threshold=0.0):
        # Assign source language based on compare between entropies.
        # For comparison with base case, threshold acts as a bias.
        wl.add_entries(
            self.donor_lng, "doculect", lambda x: "", override=True)

        for idx in wl:
            lang = wl[idx, "doculect"]
            if lang not in self.languages: continue

            # Construct iterator in order to get the initial category and
            # add the threshold (bias) to its cross-entropy.
            iter_cat = iter(self.categories)
            min_category = next(iter_cat)
            ce_category = "ce_" + min_category.lower()
            min_ce = wl[idx, ce_category] + threshold
            for category in iter_cat:
                ce_category = "ce_" + category.lower()
                if wl[idx, ce_category] < min_ce:
                    min_ce = wl[idx, ce_category]
                    min_category = category

            wl[idx, self.donor_lng] = min_category

    def calculate_cross_entropies(self, wl):
        for category in self.categories:
            #  LingPy uses lowercase column names.
            ce_category = "ce_"+category.lower()
            wl.add_entries(ce_category, "doculect", lambda x: "")

            for idx in wl:
                lang = wl[idx, "doculect"]
                if lang not in self.languages: continue

                ce = self.language_models[(lang, category)].\
                    calculate_cross_entropy(wl[idx, self.segments])
                wl[idx, ce_category] = ce

    def train(self, thresholds=None, verbose=False):
        """
        Train to optimize the threshold for the current data and approach.
        :param thresholds: thresholds to use for optimization.
        :param verbose: Boolean indicator of verbose.
        """
        # Thresholds used to assign word category below in training.
        if self.approach in ["all", "inherited"]:
            thresholds = thresholds or [
                round(i*0.2+3.0, 3) for i in range(0, 11)]
        else:
            thresholds = thresholds or [
                round(i*0.2, 3) for i in range(-5, 6)]

        self.train_markov_word_model(verbose=verbose)

        self.calculate_cross_entropies(self)

        # Address only the dominant or borrowed approach for now.
        if self.approach not in ["dominant", "borrowed"]:
            return

        # Optimization using thresholds.
        best_t, best_f = 0, 0
        for i, threshold in enumerate(thresholds):
            # Compare cross-entropies to threshold to determine source language
            # Address only source language for now; not source id.
            # self.add_entries(self.donor_id, "doculect", lambda x: "")
            self.assign_least_donor(self, threshold=threshold)

            # Now check F1 Score on train if known_donor
            if self.known_donor in self.columns:
                fs = evaluate_borrowings_fs(
                    self,
                    self.donor_lng,
                    self.known_donor,
                    self.donors)
                if fs > best_f:
                    best_t = threshold
                    best_f = fs
        self.best_value = best_t
        self.best_score = best_f
        if verbose:
            print("Train F1 Score = {0:.2f} at threshold = {1:.3f}".
                  format(best_f, best_t))

        # Restore optimal results for thresholds.
        if len(thresholds) != 1:  # Only 1 threshold
            self.assign_least_donor(self, threshold=best_t)

    def predict(self, donors, targets):
        """
        Predict borrowings for one concept.
        """
        ...

    def predict_on_wordlist(self, wordlist):
        """
        Predict for an entire wordlist.
        """
        # Uses language models including vocabularies from train.
        # Calculates entropies for test and stores depending on approach.
        self.calculate_cross_entropies(wordlist)
        self.assign_least_donor(wordlist, threshold=self.best_value)


def register(parser):
    parser.add_argument(
        "--approach",
        type=str,
        default="dominant",
        choices=["dominant", "borrowed", "all", "inherited", "each"],
        help="dominant donor (dominant), borrowed words (borrowed),"
             "all words (all), inherited words (inherited), each donor (each)."
        )
    parser.add_argument(
        "--threshold",
        default=[round(i*0.2, 3) for i in range(-5, 6)],
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
        "--direction",
        type=str,
        default="forward",
        choices=["forward", "backward"],
        help="Select language model direction."
        )
    parser.add_argument(
        "--encoding",
        type=str,
        default="ipa",
        choices=["ipa", "sca"],
        help="Encoding used for segments."
    )


def run(args):
    if args.file:
        wl = Wordlist(args.file)
        args.log.info("Construct smallest from {fl}.".format(fl=args.file))
    else:
        wl = get_our_wordlist()
        args.log.info("Construct smallest from SaBor database.")

    if args.language:
        args.language = get_language_list(args.language, args.donor)
        wl = subset_wl(wl, args.language)
        args.log.info("Subset of languages: {}".format(args.language))
    if args.encoding == "sca":
        encode_segments_as_sca(wl)

    bor = LeastCrossEntropy(wl, direction=args.direction,
                            donors=args.donor, approach=args.approach)
    bor.train(thresholds=args.threshold, verbose=True)

    args.log.info("Trained with donors {d}, approach {appr}".
                  format(d=bor.donors, appr=bor.approach))
    args.log.info("Best: threshold {thr:.2f}, F1 score {f1:.3f}".
                  format(thr=bor.best_value, f1=bor.best_score))

    full_name = "LCE-sp-predict-{appr}-{lbl}-train".\
        format(appr=bor.approach, lbl=args.label)
    file_path = our_path("store", full_name)
    bor.output("tsv", filename=file_path, prettify=False, ignore="all")

    if args.testfile:
        wl = Wordlist(args.testfile)
        args.log.info("Test least cross-entropy from {fl}.".
                      format(fl=args.testfile))

        if args.language:
            wl = subset_wl(wl, args.language)
            args.log.info("Subset of languages: {}".format(args.language))

        if args.encoding == "sca":
            encode_segments_as_sca(wl)

        bor.predict_on_wordlist(wl)

        test_f1 = evaluate_borrowings_fs(
            wl, bor.donor_lng, bor.known_donor, bor.donors)
        args.log.info("Test: threshold {thr:.2f}, F1 score {f1:.3f}".
                      format(thr=bor.best_value, f1=test_f1))
        full_name = "LCE-sp-predict-{appr}-{lbl}-test".format(
            appr=bor.approach, lbl=args.label)
        file_path = our_path("store", full_name)
        wl.output("tsv", filename=file_path, prettify=False, ignore="all")
