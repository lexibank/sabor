"""
Evaluate data.
"""

from lexibank_sabor import Dataset as SABOR
from lingpy import *
from tabulate import tabulate


def register(parser):
    parser.add_argument("--file", action="store", default="store/pw-spa-NED-0.10.tsv")
    parser.add_argument("--language", action="store", default="Spanish")

def run(args):

    wl = Wordlist(args.file)
    tp, tn, fp, fn = 0, 0, 0, 0
    for idx in wl:
        test, gold = wl[idx, "source_language"], wl[idx, "donor_language"]
        if test == gold:
            if test == args.language:
                tp += 1
            elif not test.strip():
                tn += 1
        else:
            if test == args.language:
                fp += 1
            elif gold == args.language:
                fn += 1
    table = [
            ["", "borrowed", "not borrowed", "total"],
            ["identified", tp, fp, tp+fp],
            ["not identified", fn, tn, tn+fn],
            ["total", tp+fn, fp+tn, tp+fp+tn+fn]
            ]
    print(tabulate(table))

    
