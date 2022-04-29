"""
Get borrowings from a dataset and write them to file.
"""
import collections
from lingpy import *
from pylexibank import progressbar as pb

from lexibank_sabor import Dataset as SABOR
from clldutils.clilib import Table, add_format


def sca_distance(seqA, seqB, **kw):
    """
    Shortcut for computing SCA distances from two strings.
    """

    pair = Pairwise(seqA, seqB)
    pair.align(distance=True, **kw)
    
    return pair.alignments[0][-1]


def edit_distance(seqA, seqB):
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
        donor_segments="source_value",
        donor_id="source_id",
        func=None,
        threshold=0.45,
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
    :param donor_segments: Column to which info on value of predicted donor will be written.
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
    for concept, (donor_indices, target_indices) in pb(concepts.items()):
        # hits is a dictionary with target ID as key and list of possible donor
        # candidate ids as value 
        hits = collections.defaultdict(list)
        for idxA in donor_indices:
            for idxB in target_indices:
                score = func(wordlist[idxA, segments], wordlist[idxB, segments])
                if score < threshold:
                    hits[idxB] += [(idxA, score)]
        # we sort the hits, as we can have only one donor
        for hit, pairs in hits.items():
            B[hit] = sorted(pairs, key=lambda x: x[1])[0][0]

    wordlist.add_entries(
            donor_lng, B, lambda x: wordlist[x, "doculect"] if x != 0 else "")
    wordlist.add_entries(
            donor_segments, B, lambda x: wordlist[x, segments] if x != 0 else "")
    wordlist.add_entries(
            donor_id, B, lambda x: x if x != 0 else "")


def print_borrowings(wordlist):
    """
    :param wordlist: LingPy wordlist
    :return borrowings: Count of number of borrowings.
    """
    borrowings = 0
    with Table("ID", "Language", "Concept", "Value", "Donor", "Donor_Concept",
               "Donor_Value", "WOLD_Source") as table:
        for idx in wordlist:
            if wordlist[idx, "doculect"] not in ["Spanish"]:
                if wordlist[idx, "source_language"]:
                    borrowings += 1
                    idxB = int(wordlist[idx, "source_id"])
                    table.append([
                        idx,
                        wordlist[idx, "language"],
                        wordlist[idx, "concept"],
                        wordlist[idx, "tokens"],
                        wordlist[idxB, "language"],
                        wordlist[idxB, "concept"],
                        wordlist[idxB, "tokens"],
                        wordlist[idx, "donor_language"]
                    ])
            if borrowings >= 20:
                break
    return borrowings


def register(parser):
    add_format(parser, default="simple")
    parser.add_argument(
            "--full",
            action="store_true",
            help="run the full analysis across various params"
            )


def run(args):

    SAB = SABOR()

    def get_sabor_wordlist():
        wl = Wordlist.from_cldf(
            str(SAB.cldf_dir / "cldf-metadata.json"),
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

    def run_analysis(name, donors, threshold, function, arg):
        """
        Shortcut for running the analysis.
        """
        wl = get_sabor_wordlist()
        arg.log.info("## running experiment {0}".format(name))
        pairwise_comparison(
                wl, donors, func=function, threshold=threshold)
        wl.output("tsv", filename=str(SAB.dir / "store" / name),
                  prettify=False, ignore="all")
        arg.log.info("## wrote results to file")
        arg.log.info("## found {0} borrowings".format(
            len([x for x in wl if wl[x, "source_id"]])))
        arg.log.info("---")

    if args.full:
        for name, donors in [
                ("pw-spa-por", ["Spanish", "Portuguese"]),
                ("pw-spa", ["Spanish"]), ("pw-por", ["Portuguese"])]:
            for threshold in [0.1, 0.2, 0.3, 0.4]:
                for fname, func in [
                        ("NED", edit_distance), 
                        ("SCA", sca_distance)]:
                    this_name = name + "-" + fname + "-{0:.2f}".format(threshold)
                    args.log.info("# ANALYSIS: T={0:.2f}, D={1}, F={2}".format(
                        threshold,
                        fname,
                        name))
                    run_analysis(
                            this_name,
                            donors,
                            threshold,
                            func,
                            args)
    
    else:
        wl = get_sabor_wordlist()

        args.log.info("loaded wordlist with {0} concepts and {1} languages".format(
            wl.height, wl.width))
        pairwise_comparison(wl, ["Spanish"])

        borrowings_cnt = print_borrowings(wl)
        args.log.info("Found {0} borrowings".format(borrowings_cnt))
