"""
    Functions to support performing experiments on various intruder based
    cross-family cognate and borrowed word detection.

    John E. Miller, Apr 16, 2022
"""
from pathlib import Path
import csv
from tabulate import tabulate


def report_results(results, folder, filename):
    file_path = Path(folder).joinpath(filename).as_posix()
    header = ['tp', 'tn', 'fp',  'fn', 'precision', 'recall', 'F1', 'Accuracy']
    words_table = tabulate(results, headers=header, tablefmt="pip", floatfmt=".3f")
    # with open(file_path + '.txt', 'w') as f:
    #     print(words_table, file=f)

    # Report as .tsv
    with open(file_path + '.tsv', 'w', newline='') as f:
        wrt = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        wrt.writerow(header)
        for row in results:
            wrt.writerow(row)

