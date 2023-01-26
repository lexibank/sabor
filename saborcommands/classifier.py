"""
Borrowings by Classifier
"""
from lingpy import *
from lexibank_sabor import (
        get_our_wordlist, sca_distance, edit_distance,
        evaluate_borrowings, evaluate_borrowings_fs,
        our_path,
        subset_wl, get_language_list)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from functools import partial, partialmethod
import collections
from saborcommands import (least, cognate)


def distance_by_idx(idxA, idxB, wordlist, distance=None, **kw):
    return distance(
            wordlist[idxA, "tokens"],
            wordlist[idxB, "tokens"],
            **kw)


# predefined functions
clf_sca_gl = partial(
        distance_by_idx,
        distance=sca_distance,
        mode='global')

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
    # return one_hot(int(wl[idx, 'langid'])-1, len(wl.cols), value)


class ClassifierBasedBorrowingDetection(LexStat):

    def __init__( 
            self, 
            infile,
            donors,
            clf=None,
            funcs=None,
            least_ce=None,
            cognate_cf=None,
            props=[],
            props_tar=None,
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
        self.funcs = [clf_sca_gl, clf_ned] if funcs is None else funcs
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
        self.segments = segments
        self.known_donor = known_donor

        self.best_score = 0

        self.least_ce = least_ce or []
        self.ce_objs = []
        for name in self.least_ce:
            self.ce_objs.append(least.CrossEntropyModel(
                self, self.donors,
                approach=name,
                segments=self.segments,
                known_donor=self.known_donor))

        self.cognate_cf = cognate_cf or None
        self.cognate_obj = None
        if self.cognate_cf:
            self.cognate_obj = cognate.CognateBasedBorrowing(self)

    def construct_wordlist(self, infile):
        """
        Make borrowing classifier wordlist for other file prediction.
        Required in order to reference LexStat columns and trained scorer.
        :param infile: wordlist or wordlist file reference.
        :return: ClassifierBasedBorrowingDetection object.
        """
        return ClassifierBasedBorrowingDetection(
            infile,
            self.donors,
            clf=self.clf,
            funcs=self.funcs,
            least_ce=self.least_ce,
            cognate_cf=self.cognate_cf,
            props=self.props,
            props_tar=self.props_tar,
            segments=self.segments,
            ipa=self.ipa,
            known_donor=self.known_donor
        )

    def train(self, verbose=False, log=None):
        """
        Train the classifier.
        """

        # Train cross_entropy methods.
        for ce_method in self.ce_objs:
            ce_method.train_markov_word_model(verbose=verbose)

        # set up all pairs for the training procedure
        dt, pairs, result = {}, [], []
        # dt is separated donor, target lists of indices by concept.
        for concept in self.rows:
            dt[concept] = [[], []]
            idxs = self.get_list(row=concept, flat=True)
            dt[concept][0] = [idx for idx in idxs
                              if self[idx, 'doculect'] in self.donors]
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
            log.info("Languages: {}".format(self.cols))
            log.info("Length of row = {}.".format(len(matrix[0])))
            for row in matrix[:5]: log.info(row)

        self.clf.fit(matrix, result)
        if verbose:
            log.info("Trained the classifier.")

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
        for meth in self.ce_objs:
            # Uses language models from training and infers to wl.
            row += meth.calculate_word_cross_entropies(idx_tar, wl)
        if self.cognate_obj:
            # Uses same thresholds for both train and test, and
            # clusters over current wl - whether train or test.
            # Learning is for the classifier over train data.
            row += wl.cognate_obj.\
                get_donor_target_similarity(idx_don, idx_tar)
        return row

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
        B_id = {idx: "" for idx in wordlist}
        for concept in wordlist.rows:
            idxs = wordlist.get_list(row=concept, flat=True)
            donors = [idx for idx in idxs
                      if wordlist[idx, "doculect"] in self.donors]
            targets = [idx for idx in idxs if idx not in donors]
            hits = self.predict(donors, targets, wordlist)
            for hit, pair in hits.items():
                if pair[1]:
                    B[hit] = wordlist[pair[0], 'doculect']
                    B_id[hit] = pair[0]
        wordlist.add_entries('source_language', B, lambda x: x)
        wordlist.add_entries('source_id', B_id, lambda x: x)


def register(parser):
    parser.add_argument(
        "--function",
        nargs="*",
        default=["sca", "ned"],
        choices=["sca", "ned", "sca_lo", "sca_ov"],
        help="select similarity functions for use by classifier."
    )
    parser.add_argument(
        "--least",
        nargs="*",
        default=None,
        choices=['borrowed', 'dominant'],
        help="select cross-entropy methods for use by classifier."
    )
    parser.add_argument(
        "--cognate",
        default=None,
        choices=['standard'],
        help="select cognate based method cluster arguments."
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


def run(args):
    function = {"ned": clf_ned, "sca": clf_sca_gl,
                "sca_lo": clf_sca_lo, "sca_ov": clf_sca_ov}

    if args.file:
        wl = Wordlist(args.file)
        args.log.info("Construct classifier from {fl}.".format(fl=args.file))
    else:
        wl = get_our_wordlist()
        args.log.info("Construct classifier from SaBor database.")

    if args.language:
        args.language = get_language_list(args.language, args.donor)
        wl = subset_wl(wl, args.language)
        args.log.info("Subset of languages: {}".format(args.language))

    functions = [function[key] for key in args.function]
    bor = ClassifierBasedBorrowingDetection(
        wl, donors=args.donor,
        # clf=SVC(kernel="linear", class_weight="balanced"),
        clf=SVC(kernel="linear"),
        # clf=LogisticRegression(solver='lbfgs', class_weight=None),
        funcs=functions, least_ce=args.least,
        cognate_cf=args.cognate, family="language_family")

    bor.train(verbose=True, log=args.log)
    args.log.info("Trained with donors {d}, closest {func}, "
                  "least_ce {l}, cognate_cf {c}".
                  format(d=bor.donors, func=args.function,
                         l=args.least, c=args.cognate))
    bor.predict_on_wordlist(bor)
    train_f1 = evaluate_borrowings_fs(bor, "source_language",
                                      bor.known_donor, bor.donors)
    bor.best_score = train_f1
    args.log.info("Train: F1 score {f1:.3f}".format(f1=bor.best_score))

    full_name = "CL-sp-predict-linear_svm-{}-train".format(args.label)
    file_path = our_path("store", full_name)
    columns = [column for column in bor.columns
               if not column.startswith('bor_')]
    bor.output("tsv", filename=file_path, subset=True, cols=columns,
               prettify=False, ignore="all")

    if args.testfile:
        wl = Wordlist(args.testfile)
        args.log.info("Test classifier from {fl}.".format(fl=args.testfile))

        if args.language:
            wl = subset_wl(wl, args.language)
            args.log.info("Subset of languages: {}".format(args.language))

        wl = bor.construct_wordlist(wl)
        bor.predict_on_wordlist(wl)

        args.log.info("Train: F1 score {f1:.3f}".format(f1=train_f1))
        test_f1 = evaluate_borrowings_fs(wl, "source_language",
                                         bor.known_donor, bor.donors)
        args.log.info("Test: F1 score {f1:.3f}".format(f1=test_f1))

        full_name = "CL-sp-predict-linear_svm-{}-test".format(args.label)
        file_path = our_path("store", full_name)
        wl.output("tsv", filename=file_path, prettify=False, ignore="all")
