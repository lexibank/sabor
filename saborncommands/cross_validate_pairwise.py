"""
    Perform one-shot and k-old cross-validations of Pairwise detection
    of borrowed words from names donors.

    One-shot separates wordlist into train and test datasets in the proportion
    of (k-1)/k and 1/k, determines optimal conditions for pairwise detection based
    on train data, and performs pairwise detection separately on train and test
    datasets using the optimal conditions. Report out of results includes optimal
    conditions, and detection metrics for train and test datasets.

    K-fold cross-validation performs k one-shot cross-validations with k partitions
    of wordlist into train and test datasets, where the k test partitions are
    non-overlapping and the train partitions overlap (k-2)/k of the wordlist
    each partition each of the k-folds. One-shot results are reported for each of
    the k folds.  The average and standard deviation are reported over the k folds
    as well.

    John E. Miller, May 1, 2022
"""
import tempfile
from pathlib import Path
import csv
import random
import statistics
from tabulate import tabulate
from lingpy import Wordlist
from clldutils.clilib import add_format

import saborncommands.get_pairwise_borrowings as pair
import saborncommands.run_pairwise_experiments as exp


def store_temp_data(file_path, header, data):
    with open(file_path, 'w', newline='') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerow(header)
        write.writerows(data)


def calc_start_stop(it, k, ln):
    start = int(it * ln / k)
    stop = int((it + 1) * ln / k)
    return start, stop


def get_concept_partitions(wl, k_fold, concept_name='CONCEPT'):

    concepts = wl.rows
    random.shuffle(concepts)
    # Prepare: save wordlist as .tsv and reload file as list.
    # Initial call, prepare concept list and randomize order.
    # yield loop: select entries based on concept partition, and
    # write train and test partitions to temporary files,
    # load train and test to wordlists, and return wordlists.

    with tempfile.TemporaryDirectory() as tmp:
        file_path = Path(tmp).joinpath('full-wl').as_posix()
        wl.output('tsv', filename=file_path, ignore='all', prettify=False)
        # Read in again as list.
        file_path += '.tsv'
        with open(file_path, newline='') as f:
            rdr = csv.reader(f, delimiter='\t')
            hdr = next(rdr)
            data_ = [entry for entry in rdr]
        concept_index = hdr.index(concept_name)

        # Begin yield loop of up to k iterations.
        it = 0
        while it < k_fold:
            start, stop = calc_start_stop(it, k_fold, len(concepts))
            test_concepts = concepts[start:stop]

            # Save train and test partitions.
            # Load and yield train and test wordlists.
            train, test = [], []
            for entry in data_:
                if entry[concept_index] in test_concepts:
                    test.append(entry)
                else:
                    train.append(entry)

            print("partition:", it, k_fold, start, stop, len(train), len(test))
            train_file_path = Path(tmp).joinpath('train-wl.tsv').as_posix()
            store_temp_data(train_file_path, hdr, train)
            train_wl = Wordlist(train_file_path)
            test_file_path = Path(tmp).joinpath('test-wl.tsv').as_posix()
            store_temp_data(test_file_path, hdr, test)
            test_wl = Wordlist(test_file_path)

            yield train_wl, test_wl
            it += 1


def optimize_and_test(wl_train, wl_test, function, log):
    results = exp.perform_experiment(wl_train, function, log)
    threshold, f1 = exp.get_optimal_threshold(results)
    train_result = exp.run_ana_eval_trial(wl_train, function, threshold, log)
    log.info("Train result: {}".format(train_result))

    test_result = exp.run_ana_eval_trial(wl_test, function, threshold, log)
    log.info("Test result: {}".format(test_result))
    return train_result, test_result


def calc_avg_stdev(metrics):
    # Discard the initial test or train designation.
    averages = ['Average', metrics[0][1], metrics[0][2]]
    stdevs = ['StDev', metrics[0][1], metrics[0][2]]
    for i in range(3, len(metrics[0])):
        values = [row[i] for row in metrics]
        average = statistics.mean(values)
        stdev = statistics.stdev(values)
        averages.append(average)
        stdevs.append(stdev)
    return averages, stdevs


def register(parser):
    add_format(parser, default="simple")
    parser.add_argument(
        "--cv",
        default="1-shot",
        choices=["1-shot", "k-fold"],
        help="Which cross-validation to perform - 1-shot or k=fold."
        )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Partition factor k to use for cross-validation."
    )
    parser.add_argument(
        "--function",
        default="SCA",
        choices=["SCA", "NED"],
        help="select Needleman edit distance, or sound correspondence alignment"
        )


def run(args):
    wl = pair.get_sabor_wordlist()
    # Prepare: load wordlist as .tsv file, and randomize order.
    # Yield partitions as 1-shot or k-fold.
    # it = get_partitions(wl, args.k)

    it = get_concept_partitions(wl, args.k)

    headers = ['Dataset', 'Function', 'Threshold',
              'tp', 'tn', 'fp', 'fn',
              'precision', 'recall', 'F1 score', 'accuracy']

    if args.cv in "1-shot":
        # Take first of k partitions and return train and test file_paths.
        # Run pairwise optimization and then individual trials on train and test.
        train, test = next(it)
        train_result, test_result = \
            optimize_and_test(train, test, args.function, args.log)
        metrics = [['Train', train_result[0], train_result[1]] +
                   train_result[3:],
                   ['Test', '', ''] + test_result[3:]]
        print(tabulate(metrics, headers=headers,
                       tablefmt="pip", floatfmt=".3f"))

    else:  # "k-fold"
        # For each of k-partitions, return train and test file_paths.
        # Run pairwise optimization and then individual trials on train and test
        # for each partition.
        # Calculate average and standard deviation over partitions.
        metrics = list()
        for train, test in it:
            _, test_result = \
                optimize_and_test(train, test, args.function, args.log)
            metrics.append(['Test', test_result[0], test_result[1]]
                           + test_result[3:])
        # Calculate sum, average and stdev over results.
        averages, stdevs = calc_avg_stdev(metrics)
        metrics.append([averages[0], averages[1], averages[2]] + averages[3:])
        metrics.append([stdevs[0], stdevs[1], stdevs[2]] + stdevs[3:])

        print(tabulate(metrics, headers=headers,
                       tablefmt="pip", floatfmt=".3f"))
