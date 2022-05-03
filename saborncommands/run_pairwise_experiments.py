"""
    Run experiments on pairwise to determine optimum conditions for test.
    Fit equation to results and return model parameters.

    John E. Miller, Apr 30, 2022
"""

import copy
from numpy.polynomial import Polynomial
import csv

from lexibank_sabor import Dataset as SABOR

import saborncommands.get_pairwise_borrowings as pair
import saborncommands.detail_evaluate as evl


def run_ana_eval_trial(wl, function, threshold, log):
    wl_ = copy.deepcopy(wl)
    name = "pw-spa"
    donors = ["Spanish"]

    gop = -1 if function == 'SCA' else None

    pair.run_analysis(
        wl_,
        name=name,
        donors=donors,
        threshold=threshold,
        fname=function,
        log=log,
        report=False,
        gop=gop)

    summary = evl.evaluate_detection(
        wl_,
        donors=["Spanish"],
        report=False)

    return [function, threshold, gop] + summary


def perform_experiment(wl, function, log):
    results = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        result = run_ana_eval_trial(wl, function, threshold, log)
        log.info(f"Trail result: {result}")
        results.append(result)
    store_results(function, results, log)
    return results


def get_optimal_threshold(results):
    """
    Analyze experimental results, calculate and return optimum threshold.

    :param results: Table of results from experiments.
    :return: Optimal threshold value based on experimental results.
    """

    thr_f1 = [[result[1], result[-2]] for result in results]
    thr_f1 = sorted(thr_f1, key=lambda result: result[1], reverse=True)

    poly = Polynomial.fit(x=[row[0] for row in thr_f1[:4]],
                          y=[row[1] for row in thr_f1[:4]],
                          deg=3)
    thr, f1 = poly.linspace(n=16)
    thr_f1 = sorted(zip(thr, f1), key=lambda result: result[1], reverse=True)
    opt_thr_f1 = round(thr_f1[0][0], 3), round(thr_f1[0][1], 3)
    return opt_thr_f1


def store_results(exp, results, log):
    file_path = str(SABOR().dir / "store" / ("exp_results_" + exp + ".tsv"))
    with open(file_path, 'w', newline='') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerow(['function', 'threshold', 'gop',
                        'tp', 'tn', 'fp', 'fn',
                        'precision', 'recall', 'F1 score', 'accuracy'])
        write.writerows(results)

    log.info("## Wrote results to "+file_path)


def register(parser):
    parser.add_argument(
        "--function",
        default="SCA",
        choices=["SCA", "NED"],
        help="select Needleman edit distance, or sound correspondence alignment"
        )


def run(args):
    wl = pair.get_sabor_wordlist()

    results = perform_experiment(wl, args.function, args.log)
    threshold, f1 = get_optimal_threshold(results)
    args.log.info("Estimated optimum threshold and f1 score ({:0.2f}, {:0.3f})".
                  format(threshold, f1))

    result = run_ana_eval_trial(wl, args.function, threshold, args.log)
    threshold_obs, f1_obs = result[1], result[-2]
    args.log.info("Observed threshold and f1 score ({:0.2f}, {:0.3f})".
                  format(threshold_obs, f1_obs))
