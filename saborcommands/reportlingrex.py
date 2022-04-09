"""
    Report lingrex detection results and details.

    John E. Miller, Apr 3, 2022
"""

import os
from pathlib import Path
import tempfile
import csv
import argparse
from collections import Counter, defaultdict
from tabulate import tabulate

from lingpy import *
import saborcommands.util as util


def add_donor_entries(store, infile, donors):
    filepath = Path(store).joinpath(infile).as_posix()
    wl = Wordlist(filepath)
    # Follow pattern of cluster analysis (using multple).
    # Only one threshold used so no need to iterate over multiple ids.

    etd = wl.get_etymdict(ref='autoborid')
    # external cognates doesn't seem to cross concepts so assume one concept.
    for cogid, values in etd.items():
        idxs = []
        for v in values:
            if v: idxs += v
        # Code to set single language family borids to zero.
        # Redundant since already assured by borrowing external.
        # families = [wl[idx, 'family'] for idx in idxs]
        # if len(set(families)) == 1:
        #     for idx in idxs: wl[idx, 'autoborid'] = 0
        #     print(f"Set autoborid {cogid} to 0.")
        # Code to set borids to zero if not including a donor language.
        languages = [wl[idx, 'doculect'] for idx in idxs]
        has_donor = any(donor.startswith(language)
                        for donor in donors
                        for language in languages)
        if not has_donor:
            for idx in idxs: wl[idx, 'autoborid'] = 0

    return wl


def get_table(store='store', infile=None, donors=None, any_loan=False):
    if not infile.endswith('.tsv'): infile += '.tsv'
    if donors and not any_loan:
        # Get file as wordlist and modify entries for AutoBorId
        wl = add_donor_entries(store, infile, donors=donors)
        # Store and open as flat file.
        _, filepath = tempfile.mkstemp(suffix=infile, prefix=None,
                                       dir=store, text=True)
        wl.output('tsv', filename=filepath.removesuffix('.tsv'),
                  ignore='all', prettify=False)

    else:  # No donors, so use file directly.
        filepath = Path(store).joinpath(infile).as_posix()

    # Read wordlist in flat file .tsv format from specified store and infile.
    # Dictionary input from file.
    # Convert to Boolean, Float, Int as appropriate.
    table = []
    with open(filepath, newline='') as f:
        dr = csv.DictReader(f, delimiter='\t')
        for d in dr:
            # Fixup values to use Boolean, Float, Int.
            d['BORROWED'] = None if d['BORROWED'] == '' \
                else True if d['BORROWED'] == 'True' else False
            d['BORROWED_SCORE'] = None if d['BORROWED_SCORE'] == '' \
                else float(d['BORROWED_SCORE'])
            d['AUTOCOGID'] = int(d['AUTOCOGID'])
            d['AUTOBORID'] = int(d['AUTOBORID'])
            table.append(d)

    if donors and not any_loan: os.remove(filepath)
    # Created temporary file and now removed it.

    return table


def report_basic_ids(table):
    cog_ids = [d['AUTOCOGID'] for d in table]
    bor_ids = [d['AUTOBORID'] for d in table]
    counts_cog_ids = Counter(cog_ids)
    counts_bor_ids = Counter(bor_ids)
    print(f"Word count: {sum(counts_cog_ids.values()):,}, "
          f"distinct cognates: {len(counts_cog_ids):,}.")
    del counts_bor_ids[0]
    print(f"Borid count: {sum(counts_bor_ids.values()):,}, "
          f"distinct borids: {len(counts_bor_ids):,}.")


def get_ids_table_for(table, donors=None, any_loan=False):
    family_ = [d['FAMILY'] for d in table]
    if 'LANGUAGE' in table[0]:
        language_ = [d['LANGUAGE'] for d in table]
    else:
        language_ = [d['DOCULECT'] for d in table]
    concept_ = [d['CONCEPT'] for d in table]
    # if 'CONCEPT_NAME' in table[0]:
    #     concept_name_ = [d['CONCEPT_NAME'].lower() for d in table]
    # else:
    #     concept_name_ = [c.lower() for c in concept_]
    concept_name_ = [d['CONCEPT_NAME'] for d in table]
    tokens_ = [d['TOKENS'] for d in table]
    autoborid_ = [d['AUTOBORID'] for d in table]
    loan_ = [d['BORROWED'] for d in table]
    donor_language_ = [d['DONOR_LANGUAGE'] for d in table]
    donor_value_ = [d['DONOR_VALUE'] for d in table]

    # Check that borrowing is from donor language.
    if donors and not any_loan:
        loan_ = [False if not any(donor.startswith(dl)
                                  for donor in donors) else ln
                 for dl, ln in zip(donor_language_, loan_)]

    ids_table = list(zip(family_, language_, concept_, concept_name_,
                         tokens_, autoborid_, loan_, donor_language_,
                         donor_value_))
    return ids_table


def get_borid_counts(ids_table):
    # Reports number of shared BorIds and number of shared words
    # corresponding to BorIds.
    languages = sorted(set(row[1] for row in ids_table))
    print(languages)
    borid_table = {}

    for language in languages:
        borid_set = set(row[5] for row in ids_table
                        if row[1] == language and row[5] != 0)
        borid_counts = {language: Counter() for language in languages}

        for row in ids_table:
            if row[5] in borid_set: borid_counts[row[1]][row[5]] += 1

        borid_table[language] = {lang_: [len(counter), sum(counter.values())]
                                 for lang_, counter in borid_counts.items()}

    return borid_table


def report_borid_counts(borid_counts):
    # Report borid counts shared across languages.
    keys = [key for key in borid_counts]

    for selector in range(2):  # selector: 0=concepts, 1=words.
        borid_lines = []
        for key, borids in borid_counts.items():
            borid_lines.append([key] + [borids[key_][selector] for key_ in keys])
        print()
        unit = 'concepts' if selector == 1 else 'words'
        print(f"Number of {unit} for inferred cross-family borrowing ids.")
        print(tabulate(borid_lines, headers=['Language']+keys))


def calculate_metrics(ids_table, by_fam=False):
    # [5] is borid, [6] is loan, [0] isf family, [1] is language
    pred = [0 if row[5] == 0 else 1 for row in ids_table]
    loan = [1 if row[6] else 0 for row in ids_table]
    overall = util.prf(pred, loan)

    selector = 0 if by_fam else 1  # calculate by language or by family.
    lang_group = sorted(set(row[selector] for row in ids_table))
    metrics = []
    for lang_member in lang_group:
        pred = [0 if row[5] == 0 else 1 for row in ids_table
                if row[selector] == lang_member]
        loan = [1 if row[6] else 0 for row in ids_table
                if row[selector] == lang_member]
        stats = util.prf(pred, loan)
        metrics.append([lang_member] + stats)
    metrics.append(['Overall'] + overall)

    return metrics


def report_metrics(ids_table, donors=None):
    # Drop donor words from metrics report when donors given.
    if donors:
        ids_table = [row for row in ids_table if row[1] not in donors]
    metrics = calculate_metrics(ids_table, by_fam=False)
    util.report_metrics_table(metrics, by_fam=False)
    metrics = calculate_metrics(ids_table, by_fam=True)
    util.report_metrics_table(metrics, by_fam=True)


# REPORT detail results by word entry.
def build_donor_forms_dict(ids_table, donors):
    # Construct candidate donor forms for marked candidate donor
    # for donor family and language indexed by borid.
    # Only map for the same cognate table.
    # Save form as string of tokens.
    # Also construct unmarked candidate donors indexed by concept.
    # From ids_table:
    #   family_, language_, concept_, concept_name_,
    #   tokens_, autoborid_, loan_, donor_language_,
    #   donor_value_
    # For marked use: family_, language_, autoborid_, tokens_
    # Row:              0,       1,         5,          4
    # For unmarked use: family_, language_, concepts_, tokens_
    # Row:              0,       1,         2,          4

    donor_stuff = defaultdict(lambda: defaultdict(list))
    unmarked_donor_stuff = defaultdict(lambda: defaultdict(list))

    for row in ids_table:
        if row[1] in donors:
            # Any borid > 0 interpreted as borrowed otherwise inherited.
            # For donor candidate focus, need to insert test for donor language.
            # Consider if donor not included in borid.
            # *** #
            if row[5] > 0:
                donor_stuff[row[5]][row[1]].append(row[4])
            else:  # Unmarked since no borid>0
                unmarked_donor_stuff[row[2]][row[1]].append(row[4])

    return donor_stuff, unmarked_donor_stuff


def get_words_results(ids_table, donor_forms, unmarked_donor_forms,
                      status=util.PredStatus.F):

    words = []
    family = ''
    language = ''
    concept = ''

    # Sort by language family, language, and concept.
    table_ = sorted(ids_table, key=lambda table_row:
                    (table_row[0], table_row[1], table_row[2]))

    for row in table_:
        borid_ = row[5]

        pred = 0 if borid_ == 0 else 1
        loan = row[6]
        status_ = util.assess_pred(pred, loan)

        if not util.report_assessment(status, status_): continue

        # Much effort just to put white space into report,
        # by not duplicating string values unnecessarily.

        # Do we start a new language family?
        if family != row[0]: words.append([])
        family_ = row[0] if family != row[0] else ''
        family = row[0]
        # Do we start a new language?
        if language != row[1]: words.append([])
        language_ = row[1] if language != row[1] else ''
        language = row[1]
        # Do we start a new concept?
        concept_ = row[2] if concept != row[2] else ''
        concept = row[2]

        borrowed_ = True if loan else False
        tokens_ = row[4]
        donor_language = row[7] if len(row) > 7 else ''
        donor_value = row[8] if len(row) > 8 else ''

        status_name = status_.name

        # Use concept name.
        concept_name_ = '' if not concept_ else row[3]  # concept_name

        # Add donor forms to reporting.
        # Donor forms are indexed by cogid so only available for cogid>0.
        if borid_ != 0:
            donors = donor_forms[borid_]
            # candidate donors indexed by cogid.
            marked = '*'
        else:
            donors = unmarked_donor_forms[row[2]]
            # candidate donors indexed by concept_id
            marked = ' '

        # Truncate donor_value and candidate_tokens at <= 30 characters.
        if len(donors.items()) == 0:
            words.append([family_, language_,
                          f'{concept_name_:.20}',
                          f'{tokens_:.40}',
                          donor_language,
                          f'{donor_value:20}',
                          borid_, borrowed_, status_name])
        else:
            for candidate, candidate_tokens in donors.items():
                words.append([family_, language_,
                              f'{concept_name_:.20}',
                              f'{tokens_:.40}',
                              donor_language,
                              f'{donor_value:.20}',
                              borid_, borrowed_,  status_name,
                              candidate, marked,
                              '{:.50}'.format(f'{candidate_tokens}')])
                family_ = ''
                language_ = ''
                concept_name_ = ''
                borid_ = ''
                borrowed_ = ''
                tokens_ = ''
                status_name = ''
                donor_language = ''
                donor_value = ''

    return words


def report_words_table(words, output=None, series=''):

    filename = f"lingrex-{series}{'-' if series else ''}words-status"
    file_path = Path(output).joinpath(filename).as_posix()
    header = ['Family', 'Language', 'Concept',  'Tokens',
              'Donor Language', 'Donor Value', 'BorId',
              'Borrowed', 'Status', 'Donor Candidate', 'Mark',
              'Candidate Tokens']

    words_table = tabulate(words, headers=header, tablefmt="pip")
    with open(file_path + '.txt', 'w') as f: print(words_table, file=f)


def report_words(ids_table, donors=None, status=None,
                 output=None, series=''):
    # Construct dictionary of candidate donor forms
    # of possible borrowed words for words report.
    donor_forms, unmarked_donor_forms = \
        build_donor_forms_dict(ids_table, donors=donors)

    if donors:
        ids_table = [row for row in ids_table if row[1] not in donors]

    words = get_words_results(ids_table,
                              donor_forms=donor_forms,
                              unmarked_donor_forms=unmarked_donor_forms,
                              status=status)

    report_words_table(words,
                       output=output,
                       series=series)


def run(args):
    table = get_table(infile=args.infile,
                      donors=args.donor,
                      any_loan=args.anyloan)

    report_basic_ids(table)
    ids_table = get_ids_table_for(table,
                                  donors=args.donor,
                                  any_loan=args.anyloan)

    borid_counts = get_borid_counts(ids_table)
    report_borid_counts(borid_counts)

    report_metrics(ids_table, args.donor)

    report_words(ids_table, donors=args.donor,
                 status=util.PredStatus[args.status.upper()],
                 output=args.output, series=args.series)


def register(parser):
    parser.add_argument(
        "--infile",
        type=str,
        default="lingrex-analysis"
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
    parser.add_argument(
        "--series",
        type=str,
        default='report',
        help='Filename prefix for borrowed word predictions.'
    )


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
