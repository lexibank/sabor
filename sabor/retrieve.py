"""
    Functions to retrieve ids and corresponding results from
    intermediate Bor Id and Sca Id files output by analyses.
"""

import os
from pathlib import Path
import tempfile
import csv
import re
import math
from collections import Counter
from tabulate import tabulate

from lingpy import *


def get_ids_table_for(table, ref_id=None, loc_id=None, fam_id=None,
                      donors=None, any_loan=False):
    header = ["FAMILY", "DOCULECT", "CONCEPT", "CONCEPT_NAME", "TOKENS",
              "BORROWED", "DONOR_LANGUAGE", "DONOR_VALUE"]

    ids_table = []
    for entry in table:
        ids_ = {key: entry[key] for key in header}
        if ref_id: ids_["CROSS_FAMILY_ID"] = int(entry[ref_id])
        if loc_id: ids_["LOCAL_ID"] = int(entry[loc_id])
        if fam_id: ids_["FAMILY_ID"] = entry[fam_id]
        if not any_loan and donors:
            dl = ids_["DONOR_LANGUAGE"]
            ln = ids_["BORROWED"]
            ids_["BORROWED"] = False if not \
                any(donor.startswith(dl) for donor in donors) else ln

        ids_table.append(ids_)

    return ids_table


# ===========================
# Common build table function
# ===========================
def build_table(filepath):
    # Process as normal tsv file format for wordlist.
    table = []
    with open(filepath, newline='') as f:
        dr = csv.DictReader(f, delimiter='\t')
        for d in dr:
            try:
                # Fixup values to use Boolean.
                d['BORROWED'] = None if d['BORROWED'] == '' \
                    else True if d['BORROWED'].lower() == 'true' else False
                d['BORROWED_SCORE'] = None if d['BORROWED_SCORE'] == '' \
                    else float(d['BORROWED_SCORE'])
                table.append(d)
            except AttributeError:
                # If comments in file, these will be skipped by exception.
                print(f"Skipped borrowed, tokens: {d['BORROWED']}, {d['TOKENS']}.")

    return table


def set_ref_id_zero(wl, ref_id, donors):
    etd = wl.get_etymdict(ref=ref_id)

    for cogid, values in etd.items():
        idxs = []
        for v in values:
            if v: idxs += v
        # Set ref to zero if not including a donor language.
        languages = [wl[idx, 'doculect'] for idx in idxs]
        has_donor = any(donor.startswith(language)
                        for donor in donors
                        for language in languages)
        if not has_donor:
            for idx in idxs: wl[idx, ref_id] = 0


def zero_non_donor_entries(wl, donors, id_cnt, ref_template):
    for idx in range(id_cnt):
        ref_id = ref_template.format(idx)
        set_ref_id_zero(wl, ref_id, donors)


# =============================================
# Get parameters and thresholds from table end.
# =============================================
def get_thresholds(cluster_desc):
    tex = re.compile(r"_(\d\.\d+)")
    thresholds = tex.findall(cluster_desc)
    return [float(t) for t in thresholds]


def get_params_offset(data_):
    for offset in range(-5, 0, 1):
        if str(data_[offset][0]).startswith("# Created"):
            return offset
    return 0  # Not found.


def get_parameters(store='store', infile=None):
    if not infile.endswith('.tsv'): infile += '.tsv'
    filepath = Path(store).joinpath(infile).as_posix()
    with open(filepath, newline='') as f:
        rdr = csv.reader(f, delimiter='\t')
        _ = next(rdr)
        data_ = [row for row in rdr]
        offset = get_params_offset(data_)
        parameters = [data_[idx][0] for idx in range(offset, 0)]
        thresholds = get_thresholds(parameters[-1])

    print(f"Analysis file: {filepath}")
    for parameter in parameters: print(parameter)
    print()

    return parameters, thresholds


# ===================
#  Get analysis table
# ===================
def get_ana_table(store='store', infile=None, donors=None,
                  ref_template='', ref_cnt=1, any_loan=False):
    if not infile.endswith('.tsv'): infile += '.tsv'
    filepath = Path(store).joinpath(infile).as_posix()

    if donors and not any_loan:
        wl = Wordlist(filepath)
        zero_non_donor_entries(wl, donors=donors,
                               id_cnt=ref_cnt,  # len(thresholds),
                               ref_template=ref_template)  # "SCALLID_{}")
        # Store in updated file path.
        _, filepath = tempfile.mkstemp(suffix=infile, prefix=None,
                                       dir=store, text=True)
        wl.output('tsv', filename=filepath.removesuffix('.tsv'),
                  ignore='all', prettify=False)
        # Store of wordlist already dropped final parameter rows.
        table = build_table(filepath)
        os.remove(filepath)
    else:
        # No need to zero non donor entries.
        table = build_table(filepath)

    return table


# ===============================================================
# Report on overall local, family, and cross family cognate ids.
# Cross family cognate ids are indicator of borrowing.
# ===============================================================
def get_ids_for(ids_table):
    if "LOCAL_ID" in ids_table[0]:
        local_ids = [d["LOCAL_ID"] for d in ids_table]
    else: local_ids = None
    if "FAMILY_ID" in ids_table[0]:
        family_ids = [d["FAMILY_ID"] for d in ids_table]
        expr = re.compile(r'(\d+)-')
        global_ids = [int(expr.match(fid).group(1)) for fid in family_ids]
    else: global_ids = None
    if "CROSS_FAMILY_ID" in ids_table[0]:
        cross_family_ids = [d["CROSS_FAMILY_ID"] for d in ids_table]
    else: cross_family_ids = None

    return local_ids, global_ids, cross_family_ids


def report_overall_ids(ids_table, threshold=float('NaN')):
    print()
    if threshold and not math.isnan(threshold):
        print(f"Results for threshold {threshold}.")
    local_, global_, cross_ = get_ids_for(ids_table)

    rept_lines = list()
    if local_:
        local_counts = Counter(local_)
        local_counts_gt1 = {k: v for k, v in local_counts.items() if v > 1}
        rept_lines.append(["local cognate", f"{len(local_counts):,}",
                          f"{sum(local_counts.values()):,}"])
        rept_lines.append(["local cognate multiple", f"{len(local_counts_gt1):,}",
                          f"{sum(local_counts_gt1.values()):,}"])
    if global_:
        global_counts = Counter(global_)
        global_counts_gt1 = {k: v for k, v in global_counts.items() if v > 1}
        rept_lines.append(["global cognate multiple", f"{len(global_counts_gt1):,}",
                          f"{sum(global_counts_gt1.values()):,}"])
    if cross_:
        cross_counts = Counter(cross_)
        del cross_counts[0]
        rept_lines.append(["cross family cognate", f"{len(cross_counts)}",
                          f"{sum(cross_counts.values()):,}"])
    if len(rept_lines) > 0:
        headers = ["Property", "Type", "Token"]
        print(tabulate(rept_lines, headers=headers,
                       colalign=('left', 'right', 'right')))
