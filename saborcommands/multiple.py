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
                    any_loan=False):

    # method: sca, lexstat, edit-dist, turchin
    # mode: global, local, overlap, dialign
    # cluster_method: upgma, infomap

    print(f"module {module}, method= {method}, model {model},"
          f" cluster method {cluster_method}, idtypoe {idtype},"
          f" any loan {any_loan}")

    # dataset = util.compose_wl()
    if module == 'cluster':
        wl = LexStat(dataset)
        if method == "lexstat":
            wl.get_scorer(runs=runs)
    elif module == 'partial':
        wl = Partial(dataset)
        if method == "lexstat":
            # partial scorer errors.
            wl.get_partial_scorer(runs=runs)
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
        # John: Create variable "sca_{i}", using variables "{sca_id},family".
        # John: Format the new variable with 1st part from sca_id and
        # John: second part from family.
        wl.add_entries("sca_{0}".format(i), sca_id+",family",
                       lambda x, y: str(x[y[0]]) + "-" + x[y[1]])
        # Renumber combination of cluster_id, family as integer.
        # Store in sca_{0}ID by default.
        wl.renumber("sca_{0}".format(i))
        # Get dictionary representations of cluster ids.
        # John: Seems to be dictionary of cluster ids for this threshold.
        etd = wl.get_etymdict(ref=sca_id)

        # John: Process dictionary of cluster ids.
        # Zero out cluster ids (cognate set ids) that do not cross families.
        for cogid, values in etd.items():
            # John: Construct list of row indices for this cognate set id.
            # John: What are the row indices?
            # John: Each cluster id is formed from words for the same concept.
            # John: Since clustering over words by concept.
            # John: So values is a list of indices that corresponds to target words? NO.
            # Mattis: Each row (entry) has a numeric ID in LingPy.
            # Values are a list of n items, n being the number of languages.
            # Each item contains 0 if there is no word for the concept,
            # or a list with the IDS that correspond to the cognate sets.
            # This allows having more than one item being "cognate" for the same language.

            # John: Form list of entry ids for this cluster id.
            # John: Successively extend the list by adding non-zero entry ids.
            idxs = []
            for v in values:
                if v:
                    idxs += v
            # John: Form list of language families for this cluster id.
            # John: Index must be over entries (words) for this cluster.
            # John: So we obtain corresponding language family for each word.
            families = [wl[idx, 'family'] for idx in idxs]
            # If set of just 1 family then local cognate.
            if len(set(families)) == 1:
                for idx in idxs:
                    # Set cognate id to 0 since just 1 family.
                    wl[idx, sca_id] = 0

            # John: If only checking for listed donor languages,
            # John: then test for cognate-id>0 for donor language.
            # John: This eliminates the fortuitous cross-family sca_id
            # John: when donor not included.  Could do on report side?
            if not any_loan:
                languages = [wl[idx, 'doculect'] for idx in idxs]
                has_donor = any(donor.startswith(lang)
                                for donor in donors
                                for lang in languages)
                if not has_donor:
                    for idx in idxs: wl[idx, sca_id] = 0

    filename = f"{module}{'-' if series else ''}{series}{'-' if label else ''}{label}"
    file_path = Path(store).joinpath(filename).as_posix()
    wl.output('tsv', filename=file_path, ignore='all', prettify=False)
    # Mattis: if you do not ignore = 'all', you can save the data with the
    # scorer and re-use it, so you save time!  # ignore=['scorer'] instead of ignore='all'.
    
    return filename


def register(parser):
    parser.add_argument(
        "--module",
        type=str,
        choices=["cluster", "partial"],
        default="cluster",
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
        "--anyloan",
        action="store_true",
        help='Any loan regardless of donor.'
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
    # Can run multiplereport with infile.
    if args.infile:
        rept.run(args)
        return
        
    filename = args.foreign
    wl = util.get_wordlist(filename)

    # Sub-select languages based on languages and donors arguments.
    args.language = util.get_language_all(wl) if args.language[0] == 'all' else args.language
    wl = util.select_languages(wl, languages=args.language, donors=args.donor)
    
    filename = analyze_lexstat(
        dataset=wl,
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
        any_loan=args.anyloan)

    # Report out the results.
    args.infile = filename
    rept.run(args)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
