"""
Analysis output -- some variable interpretations:
scalllid_{i} - cognate id for cognates that cross families.
sca_{i} - cognate id and language family for cognates whether or not they cross families.
sca_{i}ID - cognate id and family combination renumbered as integer.

"""
from pathlib import Path
import argparse
from lingpy import *
from lingpy.compare.partial import Partial
import saborcommands.util as util
import saborcommands.reportmultiple as rept

import lingrex
from lingrex import borrowing, cognates
from lingpy import evaluate
from clldutils.markup import Table


def analyze_lingrex(dataset,
                    module=None,
                    method='lexstat',
                    model='sca',
                    threshold=0.50,
                    runs=2000,  # during development.
                    mode='overlap',
                    cluster_method='infomap',
                    idtype='loose',
                    store='store',
                    series='common-morpheme',
                    label="",
                    donors=None,
                    any_donor_language=False
                    ):
    # ignore arguments since prototyping lingrex.
    wl = dataset

    # See paper, section "4 Results" and section "3.2 Methods".
    # Detect partial cognates:
    lingrex.borrowing.internal_cognates(
        wl,
        family='language_family',
        partial=True,
        runs=runs,
        ref="autocogids",
        method="lexstat",
        threshold=threshold,
        cluster_method=cluster_method,
        model=model)
    # Convert partial cognates into full cognates:
    lingrex.cognates.common_morpheme_cognates(
        wl,
        ref="autocogid",
        cognates="autocogids",
        morphemes="automorphemes")
    # Detect cross-family shallow cognates:
    lingrex.borrowing.external_cognates(
        wl,
        cognates="autocogid",
        ref="autoborid",
        threshold=0.3)

    # Output the evaluation:
    # p1, r1, f1 = evaluate.acd.bcubes(wl, "ucogid", "autocogid", pprint=False)
    # p2, r2, f2 = evaluate.acd.bcubes(wl, "uborid", "autoborid", pprint=False)
    # print('')
    # with Table("method", "precision", "recall", "f-score",
    #            tablefmt="simple", floatfmt=".4f") as tab:
    #     tab.append(["automated cognate detection", p1, r1, f1])
    #     tab.append(["automated borrowing detection", p2, r2, f2])
    # print('')

    filename = f"{module}{'-' if series else ''}{series}{'-' if label else ''}{label}"
    file_path = Path(store).joinpath(filename).as_posix()
    wl.output('tsv', filename=file_path, ignore='all', prettify=False)


# See lingrex/borrowing for use of different modules.
def analyze_lexstat(dataset,
                    module=None,
                    method='lexstat',
                    model='sca',
                    thresholds=None,
                    runs=1000,
                    mode='overlap',
                    cluster_method='infomap',
                    idtype='loose',
                    store='store',
                    series='analysis',
                    label="",
                    donors=None,
                    any_donor_language=True):

    # method: sca, lexstat, edit-dist, turchin
    # mode: global, local, overlap, dialign
    # cluster_method: upgma, infomap

    # dataset = util.compose_wl()
    if module == 'cluster':
        wl = LexStat(dataset)
        if method == "lexstat":
            wl.get_scorer(runs=runs, ratio=(3, 2))
    elif module == 'partial':
        wl = Partial(dataset, check=True)
        if method == "lexstat":
            # partial scorer errors.
            wl.get_partial_scorer(runs=runs, ratio=(3, 2))
    else:
        raise NameError(f"{module} not a known cluster module.")

    for i, t in enumerate(thresholds):
        print(f"Processing for threshold {t:.3f}")
        # Cluster using specified method and threshold
        # Cluster id stored in wordlist as variable scallid_{i}.
        sca_id = "scallid_{0}".format(i)
        if module == 'cluster':
            wl.cluster(method=method,
                       model=model,
                       threshold=t,
                       mode=mode,
                       cluster_method=cluster_method,
                       ref=sca_id)
        elif module == 'partial':
            sca_ids = "scallids_{0}".format(i)
            wl.partial_cluster(method=method,
                               model=model,
                               threshold=t,
                               mode=mode,
                               cluster_method=cluster_method,
                               ref=sca_ids)
            # Construct single cognate ids from lists of ids.
            wl.add_cognate_ids(sca_ids, sca_id, idtype=idtype)  # or 'strict'
            # Could also align partial cognates for output.
        else:
            raise NameError(f"{module} not a known cluster module.")

        # Add entries for cluster id combined with language family.
        # John: Formatting provided by lambda expression. Not obvious!
        # John: Create variable "sca_{i}", using variables "{sca_id}, language_family".
        # John: Format the new variable with 1st part from sca_id and
        # John: second part from language_family.
        wl.add_entries("sca_{0}".format(i), sca_id+",language_family",
                       lambda x, y: str(x[y[0]]) + "-" + x[y[1]])
        # Renumber combination of cluster_id, family as integer.
        # Store in sca_{0}ID by default.
        wl.renumber("sca_{0}".format(i))
        # Get dictionary representations of cluster ids.
        # John: Seems to be dictionary of cluster ids for this threshold.
        etd = wl.get_etymdict(ref=sca_id)

        # John: Process dictionary of cluster ids.
        # Zero out cluster ids (cognate ids) that do not cross families.
        for cogid, values in etd.items():
            # John: Construct list of row indices for this cognate id.
            # John: What are the row indices?
            # John: Each cluster id is formed from words for the same concept.
            # John: So indices correspond to target words?
            idxs = []
            for v in values:
                if v:
                    idxs += v
            # John: Form list of language families for this cluster id.
            # John: Index must be over entries (words) for this cluster.
            # John: So we obtain corresponding language family for each word.
            families = [wl[idx, 'language_family'] for idx in idxs]
            # If set of just 1 family then local cognate.
            if len(set(families)) == 1:
                for idx in idxs:
                    # Set cognate id to 0 since just 1 family.
                    wl[idx, sca_id] = 0

            # If only checking for listed donor languages
            # then test for cognate-id>0 for donor language.
            if not any_donor_language:
                languages = [wl[idx, 'doculect'] for idx in idxs]
                has_donor = any(donor.startswith(lang)
                                for donor in donors
                                for lang in languages)
                if not has_donor:
                    for idx in idxs: wl[idx, sca_id] = 0

    filename = f"{module}{'-' if series else ''}{series}{'-' if label else ''}{label}"
    file_path = Path(store).joinpath(filename).as_posix()
    wl.output('tsv', filename=file_path, ignore='all', prettify=False)
    wl.output('qlc', filename=file_path, ignore=['scorer'], prettify=False)
    
    return filename


def register(parser):
    parser.add_argument(
        "--module",
        type=str,
        choices=["cluster", "partial", "lingrex"],
        default="cluster",
        # 'lingrex' added to experiment with lingrex module.
        # clustering module predetermined if lingrex.
        help='Which clustering module to use.',
    )
    parser.add_argument(
        "--language",
        nargs="*",
        type=str,
        default=['all'],
        help="'all' or list of languages"
        #
    )
    parser.add_argument(
        "--donor",
        nargs="*",
        type=str,
        default=["Spanish", "Portuguese"],
        help='Donor language(s).',
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sca", "lexstat", "edit-dist", "turchin"],
        default="lexstat",
        help='Scoring method (default: "lexstat").',
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sca", "asjp"],
        default="sca",
        help='Sound class model to transform tokens.'
    )
    parser.add_argument(
        "--threshold",
        nargs="*",
        type=float,
        default=[0.6],
        help='Threshold(s) to use for clustering.',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["global", "local", "overlap", "dialign"],
        default="overlap",
        help='Mode used for alignment.',
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=["upgma", "infomap"],
        default="upgma",
        help='Method to use in clustering.',
    )
    parser.add_argument(
        "--idtype",
        type=str,
        choices=["loose", "strict"],
        default="loose",
        help="Manner of unifying multiple cognate ids."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2000,
        help='Number of runs for lexstat scorer.',
    )
    parser.add_argument(
        "--store",
        type=str,
        default="store",
        help='Directory to store analysis wordlist.'
    )
    parser.add_argument(
        "--series",
        type=str,
        default=""
    )
    parser.add_argument(
        "--label",
        type=str,
        default="analysis"
    )
    parser.add_argument(
        "--foreign",
        type=str,
        default=None,
        help="Filename of flat wordlist for analysis from foreign directory."
    )
    
    # Parser arguments for reporting.
    parser.add_argument(
        "--infile",
        type=str,
        default=""
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Family name to report or None if report over all families."
    )
    parser.add_argument(
        "--byfam",
        action="store_true",
        help='Report by family not individual language.'
    )
    parser.add_argument(
        "--cogid",
        action="store_true",
        help='Report only concepts and words with cross family cognate id.'
    )
    parser.add_argument(
        "--status",
        type=str,
        default='ntn',
        choices=[f"{s.name.lower()}" for s in util.PredStatus],
        # choices=["tn", "tp", "fp", "fn", "f", "t", "ntn", "all"],
        help='Code for reporting words for status.',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help='Directory to write output.'
    )


def run(args):
    # Could use just to run multiplereport.
    if args.infile:
        rept.run(args)
        return
        
    filename = args.foreign
    wl = util.get_wordlist(filename)

    # Sub-select languages based on languages and donors arguments.
    args.language = util.get_language_all(wl) if args.language[0] == 'all' else args.language
    wl = util.select_languages(wl, languages=args.language, donors=args.donor)
    
    if args.module == 'lingrex':
        analyze_lingrex(dataset=wl,
                        module=args.module,
                        runs=args.runs,
                        model=args.model,
                        threshold=args.threshold[0],
                        method=args.method,
                        cluster_method=args.cluster_method,
                        store=args.store,
                        series=args.series,
                        label=args.label)
    else:
        filename = analyze_lexstat(dataset=wl,
                        module=args.module,
                        method=args.method,
                        model=args.model,
                        thresholds=args.threshold,
                        mode=args.mode,
                        cluster_method=args.cluster_method,
                        idtype=args.idtype,
                        runs=args.runs,
                        store=args.store,
                        series=args.series,
                        label=args.label,
                        donors=args.donor,
                        any_donor_language=False)
                        
        # Report out the results.
        args.infile = filename
        rept.run(args)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
