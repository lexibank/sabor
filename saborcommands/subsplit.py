"""
Split training files into train and test for determination of best smoothing.
No need to keep concepts together since use is only for least cross-entropy.
"""
import csv
import random
from lexibank_sabor import (our_path)


def store_data(file_path, header, data):
    with open(file_path, 'w', newline='') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerow(header)
        write.writerows(data)


def split_data(in_path, train_path, test_path, k_fold):
    with open(in_path, newline='') as f:
        rdr = csv.reader(f, delimiter='\t')
        hdr = next(rdr)
        data = [entry for entry in rdr]

    random.shuffle(data)
    divider = int(round(len(data)*1/k_fold))
    print('Data:', len(data), divider, in_path)

    test = data[:divider]
    train = data[divider:]

    store_data(train_path, hdr, train)
    print('Train:', len(train), train_path)
    store_data(test_path, hdr, test)
    print('Test:', len(test), test_path)


def split_wordlists(folder, k_fold):
    it = 0
    while it < k_fold:
        in_name = "CV{k:d}-fold-{it:02d}.tsv".format(k=k_fold, it=it)
        in_path = our_path(folder, in_name)
        train_name = "CV{k:d}-fold-{it:02d}-train.tsv".format(k=k_fold, it=it)
        train_path = our_path(folder, train_name)
        test_name = "CV{k:d}-fold-{it:02d}-test.tsv".format(k=k_fold, it=it)
        test_path = our_path(folder, test_name)

        split_data(in_path, train_path, test_path, k_fold)
        it += 1


def register(parser):
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Partition factor k to use for cross-validation."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="smoothing",
        help="Folder to store train-test splits for smoothing experiment."
    )


def run(args):
    split_wordlists(args.folder, args.k)
