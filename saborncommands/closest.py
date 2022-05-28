"""
Simple Donor Search based on pairwise sequence comparison.
"""

from lexibank_sabor import (
        get_our_wordlist, our_path,
        sca_distance, edit_distance,
        simple_donor_search,
        sds_by_concept,
        evaluate_borrowings_fs,
        subset_wl,
        get_language_list
)

from lingpy import *


class SimpleDonorSearch(Wordlist):
    def __init__(
            self, 
            infile,
            donors,
            func=None,
            segments="tokens",
            known_donor="donor_language",
            **kw
            ):
        """
        Function allows to test thresholds to identify borrowings with the \
                simple_donor_search function.
        """
        Wordlist.__init__(self, infile, **kw)
        self.func = func or sca_distance
        self.donors = [donors] if isinstance(donors, str) else donors

        # Define wordlist field names.
        self.segments = segments
        self.known_donor = known_donor

        self.best_value = 0
        self.best_score = 0
        self.best_key = 'threshold'

    def train(self, thresholds=None, verbose=False):
        """
        Train the threshold on the current data.
        """
        thresholds = thresholds or [i*0.05 for i in range(1, 20)]
        
        # calculate distances between all pairs, only once, afterwards make a
        # function out of it, so we can pass this to the simple_donor_search
        # function
        D = {}
        for concept in self.concepts:
            idxs = self.get_list(row=concept, flat=True)
            donors = [idx for idx in idxs
                      if self[idx, 'doculect'] in self.donors]
            targets = [idx for idx in idxs if idx not in donors]
            for idxA in donors:
                for idxB in targets:
                    tksA, tksB = self[idxA, self.segments], self[
                            idxB, self.segments]
                    D[str(tksA), str(tksB)] = self.func(tksA, tksB)
        new_func = lambda x, y: D[str(x), str(y)]
        if verbose: print("computed distances")

        best_t, best_f = 0, 0
        for i, threshold in enumerate(thresholds):
            if verbose: print("analyzing {0:.2f}".format(threshold))
            tidx = "t_"+str(i+1)
            simple_donor_search(
                    self,
                    self.donors,
                    concept=self._row_name,
                    segments="tokens",
                    donor_lng=tidx+"_lng",
                    donor_id=tidx+"_id",
                    func=new_func,
                    threshold=threshold
                    )
            if self.known_donor in self.columns:
                # Calculate F score if known donor column.
                fs = evaluate_borrowings_fs(
                        self,
                        tidx+"_lng",
                        self.known_donor,
                        self.donors)
                if fs > best_f:
                    best_t = threshold
                    best_f = fs
            if verbose: print("... {0:.2f}".format(fs))
        self.best_value = best_t
        self.best_score = best_f

    def predict(self, donors, targets):
        """
        Predict borrowings for one concept.
        """
        return sds_by_concept(donors, targets, self.func, self.best_value)

    def predict_on_wordlist(
            self, wordlist, donor_lng="source_language", 
            donor_id="source_id"):
        """
        Predict for an entire wordlist.
        """
        simple_donor_search(
                wordlist,
                self.donors,
                concept=self._row_name,
                segments=self.segments,
                donor_lng=donor_lng,
                donor_id=donor_id,
                func=self.func,
                threshold=self.best_value)


def register(parser):
    parser.add_argument(
        "--function",
        default="SCA",
        choices=["SCA", "NED"],
        help="select Needleman edit distance, or sound correspondence alignment."
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


def run(args):
    function = {"NED": edit_distance,
                "SCA": sca_distance}[args.function]

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

    bor = SimpleDonorSearch(wl, func=function, donors=args.donor)

    bor.train(thresholds=args.threshold)

    args.log.info("Trained with donors {d}, function {func}".
                  format(d=bor.donors, func=bor.func.__name__))
    args.log.info("Best: threshold {thr:.2f}, F1 score {f1:.3f}".
                  format(thr=bor.best_value, f1=bor.best_score))

    args.log.info("Predict with  threshold {t}.".format(t=bor.best_value))
    bor.predict_on_wordlist(bor)
    full_name = "CM-sp-predict-{func}-{thr:.2f}-train".format(
        func=bor.func.__name__, thr=bor.best_value)
    file_path = our_path("store", full_name)
    columns = [column for column in bor.columns
               if not column.startswith('t_')]
    bor.output("tsv", filename=file_path, subset=True, cols=columns,
               prettify=False, ignore="all")

    if args.testfile:
        wl = Wordlist(args.testfile)
        args.log.info("Test closest from {fl}.".format(fl=args.testfile))

        if args.language:
            wl = subset_wl(wl, args.language)
            args.log.info("Subset of languages: {}".format(args.language))

        bor.predict_on_wordlist(wl)
        args.log.info("Evaluation:" + str(evaluate_borrowings_fs(
            wl,
            "source_language",
            bor.known_donor,
            bor.donors)))
        full_name = "CM-sp-predict-{func}-{thr:.2f}-test".format(
            func=bor.func.__name__, thr=bor.best_value)
        file_path = our_path("store", full_name)
        wl.output("tsv", filename=file_path, prettify=False, ignore="all")
