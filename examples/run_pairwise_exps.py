"""
    Run series of experiments on Pairwise.
"""
from pathlib import Path
import argparse
import re
from csvw.dsv import UnicodeDictReader
import csv
from saborcommands import util, pairwise


def run_experiments_pairwise(languages, donors, any_loan, model, mode, gop, thresholds,
                             pbs, min_len, series, scripts, output, filename=None):
    """
    Experiment configurations are read from a file and stored in a list of dictionaries.
    Experiments are run over the given list of languages with given donors.
    Series qualifies the set of experiments.
    All information necessary to perform each experiment is given in the configuration.

    :param languages: List of languages given as text strings or the list ['all'] for all languages.
    :param donors: List of donor languages for pairwise matching.
    :param any_loan: Whether just donor (false) or any loan (true) for assessment report.
    :param model: Sound model used by Pairwise similarity or distance measure.
    :param mode: Mode for aligning sequences ['global', 'overlap', 'local'].
    :param gop: Gap open penalty.
    :param thresholds: List of thresholds to test whether alignment is meaningful.
    :param pbs: Indicator of whether to use positional scoring.
    :param series: Qualifying name for experimental series.
    :param scripts: Folder to read experiment from.
    :param output: Folder to write results to.
    :return: List of results corresponding to experiments.
    """

    print(f"Pairwise arguments: donors {donors}, any loan {any_loan}, "
          f"model {model}, mode {mode}, gop {gop}, thresholds {thresholds}, "
          f"pbs {pbs}, series {series}, scripts {scripts}, output {output}.")

    exps = []
    file_base = "pairwise-doe"

    file_path = Path(scripts).joinpath(f"{file_base}-{series}").as_posix()
    # with UnicodeDictReader(file_path + ".tsv", delimiter="\t") as rdr:
    with open(file_path + ".tsv", newline='') as f:
        rdr = csv.DictReader(f, dialect=csv.excel_tab)
        for row in rdr:
            exps.append(row)
            # print(row)

    results = []
    for expid, config in enumerate(exps, start=1):
        if "model" not in config: config["model"] = model
        if "mode" not in config: config["mode"] = mode
        if "gop" not in config: config["gop"] = gop
        if "pbs" not in config: config["pbs"] = pbs
        if "min_len" not in config: config["min_len"] = min_len
        # Translate text to float for numeric fields.
        if config.get('gop'): config['gop'] = float(config['gop'])
        if config.get('scale'): config['scale'] = float(config['scale'])
        if config.get('factor'): config['factor'] = float(config['factor'])
        if "threshold" not in config:
            config["threshold"] = thresholds
        else:
            # Make sure thresholds are formatted as list.
            thresholds = config["threshold"]
            thresholds_ = [float(item) for item in re.split(r'\[|\]|. ', thresholds) if item]
            print(f"interpreted thresholds: {thresholds_}")
            config["threshold"] = thresholds_

        if pbs:
            if config.get('min_len'): config['min_len'] = int(config['min_len'])
            if config.get('wt_fn'): config['wt_fn'] = float(config['wt_fn'])

        if pbs and ('wt_C' in config or 'wt_c' in config or
                    'wt_V' in config or 'wt_v' in config):
            # Construct weights and set in config if given.
            weights = dict()
            weights['C'] = float(config['wt_C']) if 'wt_C' in config else 1.0
            weights['c'] = float(config['wt_c']) if 'wt_c' in config else 0.8
            weights['V'] = float(config['wt_V']) if 'wt_V' in config else 0.6
            weights['v'] = float(config['wt_v']) if 'wt_v' in config else 0.4
            weights['_'] = float(config['wt__']) if 'wt__' in config else 0.0
            config['weights'] = weights

        print(config)
        result = pairwise.get_total_run_result(languages=languages, donors=donors,
                                               any_loan=any_loan, config=config,
                                               filename=filename)
        print('***', result, '***')
        results.extend(result)

    # If each experiment were very costly, would store to file after each experiment.
    # Since not costly for pairwise, this seems OK.
    util.report_results(results, output, f"{file_base}-{series}-results")


def register(parser):
    parser.add_argument(
        "--language",
        nargs="*",
        type=str,
        default=['all'],
        help="'all' or list of languages."
        #
    )
    parser.add_argument(
        "--donor",
        nargs="*",
        type=str,
        default=["Spanish", "Portuguese"],
        # We don't distinguish between LA, BR,EU versions.
        # Whichever is used to set to Spanish or Portuguese orthography filename.
        help='Donor language(s).',
    )
    parser.add_argument(
        "--anyloan",
        action="store_true",
        help='Any loan regardless of donor.'
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sca", "asjp"],
        default="sca",
        help='Sound class model to transform tokens.'
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="global",
        choices=["global", "local", "overlap"],
        help='Alignment mode.',
    )
    parser.add_argument(
        "--gop",
        type=float,
        default=-2.0,
        help='Gap open penalty.'
    )
    parser.add_argument(
        "--threshold",
        nargs="*",
        type=float,
        default=None,
        help='Threshold(s) to use with pairwise alignment method.',
    )
    parser.add_argument(
        "--pbs",
        action="store_true",
        help='Use positional based scoring.'
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help='Minimum length of match for position based scoring.'
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help='Filename qualifier for experiments.'
    )
    parser.add_argument(
        "--scripts",
        type=str,
        default="exp-scripts",
        help='Directory to get experiments from.'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp-results",
        help='Directory to write results to.'
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Foreign filename."
    )
    

if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    args = parser_.parse_args()

    run_experiments_pairwise(languages=args.language,
                             donors=args.donor,
                             any_loan=args.anyloan,
                             model=args.model,
                             mode=args.mode,
                             gop=args.gop,
                             thresholds=args.threshold,
                             pbs=args.pbs,
                             min_len=args.min_len,
                             series=args.series,
                             scripts=args.scripts, 
                             output=args.output,
                             filename=args.filename)
