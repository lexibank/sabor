"""
    Partition wordlists into k-fold train and test files for use in cross-validations.

    K-fold cross-validation permits an application to perform k one-shot
    cross-validations with k partitions of wordlist into train and test datasets,
    where the k test partitions are non-overlapping and the train partitions
    overlap (k-2)/k of the wordlist each partition each of the k-folds.
    Results can be reported for each of k folds an overall.

    Since the partition can be used over long term, this permits benchmarking various
    methods against the same train and test k-fold dataset.

"""
import tempfile
from pathlib import Path
import csv
import random
from tabulate import tabulate
from lingpy import Wordlist
from lexibank_sabor import (get_our_wordlist, our_path)


def get_train_test_data(k_fold, folder):
    # Yield train and test datasets from folder based on k_fold naming.
    it = 0
    while it < k_fold:
        full_name = "CV{k:d}-fold-{it:02d}-train.tsv".format(k=k_fold, it=it)
        file_path = our_path(folder, full_name)
        train_wl = Wordlist(file_path)
        full_name = "CV{k:d}-fold-{it:02d}-test.tsv".format(k=k_fold, it=it)
        file_path = our_path(folder, full_name)
        test_wl = Wordlist(file_path)

        yield it, train_wl, test_wl
        it += 1


def store_temp_data(file_path, header, data):
    with open(file_path, 'w', newline='') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerow(header)
        write.writerows(data)


def calc_start_stop(it, k, ln):
    start = int(it * ln / k)
    stop = int((it + 1) * ln / k)
    return start, stop


def split_wordlist(wl, k_fold, folder, concept_name='CONCEPT'):

    concepts = wl.rows
    concept_cnt = wl.height
    random.shuffle(concepts)
    # Prepare: save wordlist as .tsv and reload file as list.
    # Initial call, prepare concept list and randomize order.
    # Loop: select entries based on concept partition,
    # write train and test partitions to temporary files,
    # load train and test to wordlists, and store wordlists
    # in folder.

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

        partitions = []
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

            print("{k:d}-fold split, partition {it:2d}, "
                  "test: {tstc:d} concepts, {tste:d} entries; "
                  "train: {trnc:d} concepts, {trne:d} entries".
                  format(k=k_fold, it=it, tstc=stop-start, tste=len(test),
                         trnc=concept_cnt-stop+start, trne=len(train)))
            partitions.append([k_fold, it, stop-start, len(test),
                               concept_cnt-stop+start, len(train)])

            # Redundant to store as list and then read and store as wordlist,
            # But it assures that wordlist is actually readable.
            train_file_path = Path(tmp).joinpath('train-wl.tsv').as_posix()
            store_temp_data(train_file_path, hdr, train)
            train_wl = Wordlist(train_file_path)
            test_file_path = Path(tmp).joinpath('test-wl.tsv').as_posix()
            store_temp_data(test_file_path, hdr, test)
            test_wl = Wordlist(test_file_path)

            # Save wordlist files into folder.
            full_name = "CV{k:d}-fold-{it:02d}-train".format(k=k_fold, it=it)
            # file_path = str(SABOR().dir / folder / full_name)
            file_path = our_path(folder, full_name)
            train_wl.output("tsv", filename=file_path,
                            prettify=False, ignore="all")
            full_name = "CV{k:d}-fold-{it:02d}-test".format(k=k_fold, it=it)
            # file_path = str(SABOR().dir / folder / full_name)
            file_path = our_path(folder, full_name)
            test_wl.output("tsv", filename=file_path,
                           prettify=False, ignore="all")

            it += 1

        return partitions


def register(parser):
    # add_format(parser, default="simple")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Partition factor k to use for cross-validation."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="splits",
        help="Folder to store train-test splits for k-fold cross-validations."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Demonstrate loading of train-test datasets in sequence."
        )


def run(args):
    wl = get_our_wordlist()
    if args.test:
        print("* Reporting {k} train-test splits *".format(k=args.k))
        summary = []
        for it, train_wl, test_wl in get_train_test_data(args.k, args.folder):
            summary.append([it, test_wl.height, len(test_wl),
                            train_wl.height, len(train_wl)])
        print(tabulate(summary, headers=[
            "partition", "test concepts", "test entries",
            "train concepts", "train entries"]))

    else:
        partitions = split_wordlist(wl, args.k, args.folder)
        print("* Constructed {k} train-test splits *".format(k=args.k))
        print(tabulate(partitions, headers=[
            "k-fold", "partition", "# test concepts",
            "# test entries", "# train concepts", "# train entries"]))
