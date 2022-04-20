"""
    Run series of cluster or partial experiments using lexstat method.

"""

import argparse
from csvw.dsv import UnicodeDictReader
from saborcommands import analyzecluster, evaluatecluster
from sabor import experiment
import sabor.accessdb as adb


def get_wordlist(filename, donors):
    wl = adb.get_wordlist(filename)
    languages = adb.get_language_all(wl)
    return adb.select_languages(wl, languages=languages, donors=donors)


def run_experiments(args):
    wl = get_wordlist(args.foreign, args.donor)

    exps = []
    with UnicodeDictReader(f"exp-scripts/{args.module}-{args.series}.tsv",
                           delimiter="\t") as rdr:
        for row in rdr:  exps.append(row)

    results = []
    for exp in exps:
        # Not a problem if we include extra arguments for runs, and idtype.
        analyzecluster.analyze_lexstat(
            dataset=wl,
            module=args.module,
            method=args.method,
            model=args.model,
            runs=int(exp["runs"]) if "runs" in exp else 2000,  # default to 2000 runs
            thresholds=[float(exp["threshold"])],
            mode=exp["mode"] if "mode" in exp else "overlap",  # default to overlap
            cluster_method=exp["cluster_method"] if "cluster_method" in exp else "upgma",
            idtype=exp["idtype"] if "idtype" in exp else "strict",  # default to strict
            store="store",
            series=args.series,
            label=exp["id"])

        infile = f"{args.module}-{args.series}-{exp['id']}"
        result = evaluatecluster.get_exp_result(
            store=args.store, infile=infile, family=args.family,
            donors=args.donor, index=0, any_loan=args.any)
        print('***', result, '***')
        results.append(result)

    experiment.report_results(
        results, "exp-results",
        f"{args.module}-{args.series}-results")


def register(parser):
    parser.add_argument(
        "--module",
        type=str,
        choices=["cluster", "partial"],
        default="cluster",
        help='Which clustering module to use.',
    )
    parser.add_argument(
        "--donor",
        nargs="*",
        type=str,
        default=["Spanish", "Portuguese"],
        help='Donor language(s).',
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Family to report or None if report over all families."
    )
    parser.add_argument(
        "--any",
        action="store_true",
        help='Any loan regardless of donor.'
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sca", "lexstat", "edit-dist"],
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
        "--series",
        type=str,
        default='doe',
        help='Filename qualifier for experiments.'
    )
    parser.add_argument(
        "--store",
        type=str,
        default="store",
        help='Directory to store analysis wordlist.'
    )
    parser.add_argument(
        "--foreign",
        type=str,
        default=None,
        help="Filename of flat wordlist for analysis from foreign directory."
    )


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run_experiments(parser_.parse_args())
