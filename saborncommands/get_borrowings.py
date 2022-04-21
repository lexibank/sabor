"""
Get borrowings from a dataset and write them to file.
"""
import collections
from lingpy import *

from lexibank_sabor import Dataset as SABOR
from tabulate import tabulate


def sca_distance(seqA, seqB, **kw):
    """
    Shortcut for computing SCA distances from two strings.
    """

    pair = Pairwise(seqA, seqB)
    pair.align(distance=True, **kw)
    
    return pair.alignments[0][-1]


def pairwise_comparison(
        wordlist,
        donors,
        concept="concept",
        donor_lng="source_language",
        donor_val="source_value",
        donor_id="source_id",
        func=None,
        threshold=0.45,
        **kw
        ):
    """
    Find borrowings by carrying out a pairwise comparison of donor and target words.

    :param wordlist: The LingPy wordlist.
    :param donors: The donor languages, passed as a list.
    :param concept: The column in which concept information is given in the
      wordlist.
    :param donor_lng: The column to which information on donor languages will
      be written (defaults to "source_language").
    :param donor_val: Column to which info on value of donor will be written.
    :param donor_id: Column to which we write information on the ID of the
      donor.
    :param func: A function comparing two sequences and returning a distance
      score (defaults to sca_distance).
    :param threshold: The threshold, at which we recognize a word as being
      borrowed.
    """
    func = func or sca_distance

    # get concept slots from the data (in case we use broader concepts by clics
    # communities), we essentially already split data in donor indices and
    # target indices by putting them in a list
    concepts = {concept: [[], []] for concept in set(
        [wordlist[idx, concept] for idx in wordlist])}
    for idx in wordlist:
        if wordlist[idx, "doculect"] in donors:
            concepts[wordlist[idx, concept]][0] += [idx]
        else:
            concepts[wordlist[idx, concept]][1] += [idx]

    # iterate over concepts and identify potential borrowings
    B = {idx: 0 for idx in wordlist}
    for concept, (donors, targets) in concepts.items():
        # hits is a dictionary with target ID as key and list of possible donor
        # candidate ids as value 
        hits = collections.defaultdict(list)
        for idxA in donors:
            for idxB in targets:
                score = func(wordlist[idxA, "tokens"], wordlist[idxB, "tokens"])
                if score < threshold:
                    hits[idxB] += [(idxA, score)]
        # we sort the hits, as we can have only one donor
        for hit, pairs in hits.items():
            B[hit] = sorted(pairs, key=lambda x: x[1])[0][0]
    wordlist.add_entries(
            donor_lng, B, lambda x: wordlist[x, "doculect"] if x != 0 else "")
    wordlist.add_entries(
            donor_val, B, lambda x: wordlist[x, "tokens"] if x != 0 else "")
    wordlist.add_entries(
            donor_id, B, lambda x: x if x != 0 else "")

    
def run(args):

    sabor = SABOR()
    wl = Wordlist.from_cldf(
            str(sabor / "cldf" / "cldf-metadata.json"),
            columns = [
                "doculect", "concept", "value", "form", "tokens", 
                "donor_language"]
            )
    pairwise_comparison(wl)
    table = []
    for idx in wl:
        if wl[idx, "source_language"]:
            idxB = int(wl[idx, "source_id"])
            table += [[
                idx, 
                wl[idx, "language"], 
                wl[idx, "concept"],
                wl[idx, "segments"],
                wl[idxB, "language"],
                wl[idxB, "concept"],
                wl[idxB, "segments"]
                ]]
    print("Found {0} borrowings".format(len(table)))
    print(tabulate(table[:10], headers=[
        "ID", "Language", "Concept", "Value", "Donor", "Donor_Concept",
        "Donor_Value"], tablefmt="pipe"))


        
    
