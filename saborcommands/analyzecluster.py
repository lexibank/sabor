"""
    Analyze language forms by cluster or partial cluster analysis.
    Output -- local, family, and cross-family annotation of cluster ids.
    scalllid_{i} - int cognate id for cognates that cross families.
    sca_{i} - cognate id and language family for cognates whether or not they cross families.
    sca_{i}ID - int cognate ids assigned within families.

    John E. Miller, Apr 14, 2022
    Original by Johan-Mattis List.
"""
from pathlib import Path
import argparse
from lingpy import *
from lingpy.compare.partial import Partial

import saborcommands.evaluatecluster as evaluate
import sabor.arguments as argfns
import sabor.accessdb as adb


def analyze_lexstat(dataset,
                    module=None,
                    method='lexstat',  # sca, lexstat
                    model='sca',
                    thresholds=None,
                    runs=2000,
                    mode='overlap',  # global, local, overlap
                    cluster_method='infomap',  # upgma, infomap
                    idtype='loose',  # loose, strict
                    store='store',
                    series='analysis',
                    label=""):

    print(f"module {module}, method= {method}, model {model},"
          f" cluster method {cluster_method}, idtypoe {idtype}")

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
        # Cluster id stored in wordlist as scallid_{i}.
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
            # Construct cognate ids from lists of ids.
            wl.add_cognate_ids(sca_ids, sca_id, idtype=idtype)
        else:
            raise NameError(f"{module} not a known cluster module.")

        # Add entries for cluster id combined with language family.
        wl.add_entries("sca_{0}".format(i), sca_id+",family",
                       lambda x, y: str(x[y[0]]) + "-" + x[y[1]])
        # Renumber combination of cluster_id, family as integer.
        # Store in sca_{0}ID by default.
        wl.renumber("sca_{0}".format(i))
        # Get dictionary representations of cluster ids.
        etd = wl.get_etymdict(ref=sca_id)

        # Zero out cluster ids (cognate set ids) that do not cross families.
        for cogid, values in etd.items():
            idxs = []
            for v in values:
                if v: idxs += v
            families = [wl[idx, 'family'] for idx in idxs]
            # If set of just 1 family then local cognate.
            if len(set(families)) == 1:
                for idx in idxs:
                    # Set cognate id to 0 since just 1 family.
                    wl[idx, sca_id] = 0

    filename = f"{module}{'-' if series else ''}{series}{'-' if label else ''}{label}"
    file_path = Path(store).joinpath(filename).as_posix()
    wl.output('tsv', filename=file_path, ignore='all', prettify=False)

    return filename


def run(args):
    filename = args.foreign
    wl = adb.get_wordlist(filename)

    # Sub-select languages based on languages and donors arguments.
    args.language = adb.get_language_all(wl) \
        if args.language[0] == 'all' else args.language
    wl = adb.select_languages(wl, languages=args.language, donors=args.donor)

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
        label=args.label)

    # Report out the results.
    args.infile = filename
    evaluate.run(args)


def register(parser):
    argfns.register_common(parser)
    argfns.register_analysis(parser)

    parser.add_argument(
        "--module",
        type=str,
        choices=["cluster", "partial"],
        default="cluster",
        help='Which clustering module to use.',
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
        "--idtype",
        type=str,
        choices=["loose", "strict"],
        default="loose",
        help="Manner of unifying multiple cognate ids."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Used by experimental design module."
    )
    parser.add_argument(
        "--foreign",
        type=str,
        default=None,
        help="Filename of flat wordlist for analysis from foreign directory."
    )

    argfns.register_evaluate(parser)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
