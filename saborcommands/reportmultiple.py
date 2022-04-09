"""
    Report on cognates identified by Lingpy clustering methods.

    John E. Miller, Aug 21, 2021
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import pandas as pd
from tabulate import tabulate
import saborcommands.util as util


def get_table(store='store', infile=None):
    # Read Lexstat wordlist in flat file .tsv format from specified store and infile.
    if not infile.endswith('.tsv'):
        infile += '.tsv'
    filepath = Path(store).joinpath(infile).as_posix()
    # Uses Pandas for simple processing step.
    table = pd.read_csv(filepath, sep='\t')
    parameters = table.iloc[-3:]['ID'].values
    # Check to see if #Created on first line and so valid parameters list.
    if str(parameters[0]).startswith("# Created"):
        offset = -3
    elif str(parameters[1]).startswith("# Created"):
        offset = -2
    else:  # Assume that at least a 'Created' parameter.
        offset = -1
    parameters = table.iloc[offset:]['ID'].values

    # Change missing to '' for donor_language and donor_value
    table.DONOR_LANGUAGE = table.DONOR_LANGUAGE.fillna('')
    table.DONOR_VALUE = table.DONOR_VALUE.fillna('')

    print("Analysis file details:")
    for parameter in parameters: print(parameter)
    print()
    table.drop(table.index[offset:], inplace=True)
    print(f"{len(table)} words in {filepath}.")
    return table, parameters


def get_cogids_for(table, index=0):
    # Get local, local>1, global, global>1 for series index.
    local_label = f"SCA_{index}ID"
    global_family_label = f"SCA_{index}"
    global_gt1_label = f"SCALLID_{index}"
    local_series = table[local_label].astype('int')
    global_gt1_series = table[global_gt1_label].astype('int')
    global_family_series = table[global_family_label]
    global_series = global_family_series.str.extract(r'(\d+)\-')[0].astype('int')
    return local_series, global_series, global_gt1_series


def report_basic(table, index=0):
    local_, global_, global_gt1_ = get_cogids_for(table, index)
    
    counts_local_cognates = local_.value_counts()
    print(f"Number of words: {sum(counts_local_cognates)}, " +
          f"number of distinct local cognates: {len(counts_local_cognates)}")
    counts_local_cognates_gt1 = counts_local_cognates[counts_local_cognates.values > 1]
    print(f"Number of words for >1 local cognates: {sum(counts_local_cognates_gt1)}, " +
          f"number of distinct >1 local cognates: {len(counts_local_cognates_gt1)}")

    counts_global_cognates = global_.value_counts()
    print(f"Number of words: {sum(counts_global_cognates)}, " +
          f"number distinct global cognates: {len(counts_global_cognates)}")
    counts_global_cognates_gt1 = global_gt1_.value_counts()
    # Drop the 0 cognate as it is not a family global cognate.
    counts_global_cognates_gt1 = counts_global_cognates_gt1[counts_global_cognates_gt1.index != 0]
    print(f"Number of words for >1 family global cognates: {sum(counts_global_cognates_gt1)}, " +
          f"number of distinct >1 family global cognates: {len(counts_global_cognates_gt1)}")


def get_cogids_table_for(table, index=0, donors=None, any_loan=False):
    # Table is from Pandas, so we use column names in the code.
    family_ = list(table.FAMILY.values)
    language_ = list(table.DOCULECT.values)

    concepts_ = list(table.CONCEPT.values)
    # Use concept in lower case if concept_name not defined.
    if 'CONCEPT_NAME' in table.columns:
        concept_names_ = [val.lower() for val in table.CONCEPT_NAME.values]
    else:
        concept_names_ = [val.lower() for val in table.CONCEPT.values]

    tokens_ = list(table.TOKENS.values)
    local_, global_, global_gt1_ = get_cogids_for(table, index)
    local_ = list(local_.values)
    global_ = list(global_.values)
    global_gt1_ = list(global_gt1_.values)


    # From WOLD with donor language and value.
    loan_ = list(table.BORROWED.values)
    donor_language_ = list(table.DONOR_LANGUAGE.values)
    donor_value_ = list(table.DONOR_VALUE.values)

    # Focus on borrowings from donor languages only. ***
    # With analyze we took into account prediction using cogid.
    # Here we take into account whether loans are from donors in list.
    # Assuming donors are given.
    if not any_loan and donors:
        loan_ = [False if not any(donor.startswith(dl) for donor in donors) else ln
                    for dl, ln in zip(donor_language_, loan_)]

    cogids_table = list(zip(family_, language_,
                            local_, global_, global_gt1_,
                            loan_, tokens_,
                            concepts_, concept_names_,
                            donor_language_, donor_value_))
    return cogids_table


def get_cogids_by_family(table, family=None, byfam=False):
    # Table is intermediate result from get_cogids_table_for
    # lu refers to language unit - language or family.
    if family:
        # Filter on family.
        table = [row for row in table if row[0] == family]
    lu_idx = 0 if byfam else 1  # 0 is idx for family, 1 for language.
    # List of languages or language families depending on lu_idx.
    lu_unit_set = sorted(set(row[lu_idx] for row in table))
    lu_global_cognates_gt1 = {}

    # global_gt1 in [4] of table.
    # for each language or language family in set.
    for lu in lu_unit_set:
        # Get set of cross-family cogids for this language or language family.
        cognates_gt1 = set(row[4] for row in table
                           if row[lu_idx] == lu and row[4] != 0)

        # Construct dictionary of counters for each language or language family.
        cognates_gt1_ = {lu_: Counter() for lu_ in lu_unit_set}
        # Process all the individual entries from the cogids table.
        for row in table:
            if row[4] in cognates_gt1:
                # Count this entry if a cross-family cogid in the list
                # for this language or language family.
                # Create count for this cogid for this language or family.
                # Add to count for this cogid for this language or family
                #   for each qualified entry.
                # lu_idx indexes into either the language or family column.
                # *** could have just used 'lu' here!!!!
                cognates_gt1_[row[lu_idx]][row[4]] += 1

        # Condense detail down to numbers of cross-family concepts and words,
        # shared with each other language unit.
        lu_global_cognates_gt1[lu] = {lu_: [len(counter), sum(counter.values())]
                                      for lu_, counter in cognates_gt1_.items()}

    return lu_global_cognates_gt1


def report_cogids_table(global_cognates, selector=0, threshold=None):
    # Display as table.
    # Selector 0 : cognates, 1: words
    # lu == language unit -- language, language_family
    cognates_table = []
    lu_keys = [key for key in global_cognates.keys()]

    # For each language or language family (lu), report on number of cognate concepts
    # number of cognate words shared with another language unit.
    # Selector indexes into concept (0) or word (1) counts.
    for lu, lu_cognates in global_cognates.items():
        cognates_table.append([lu] + [lu_cognates[lu_][selector] for lu_ in lu_keys])

    print()
    unit = 'Concepts' if not selector else 'Words'
    print(f"Number of {unit} for inferred cross-family cognate ids at threshold: {threshold:0.3f}.")
    header0 = 'Language ' + unit
    print(tabulate(cognates_table, headers=[header0] + lu_keys))


def report_cogids_by_family(cogids_table, family=None, byfam=False, threshold=None):
    lu_cogids_gt1 = get_cogids_by_family(cogids_table, family=family, byfam=byfam)
    report_cogids_table(lu_cogids_gt1, selector=0, threshold=threshold)
    report_cogids_table(lu_cogids_gt1, selector=1, threshold=threshold)


def get_language_unit_table(table, family=None, byfam=False, donors=None):
    # Works with languages from single family or over families.
    # Exclude donors from the calculation.
    table = [row for row in table if row[1] not in donors]

    # lu refers to language unit - language or family.
    if family:  # Select on family.
        # Filter on family
        table = [row for row in table if row[0] == family]
        
    lu_idx = 0 if byfam else 1
    lu_units = sorted(set([row[lu_idx] for row in table]))
    return table, lu_units, lu_idx


def get_words_results(table, donor_forms, unmarked_donor_forms,
                      require_cogid=False, status=util.PredStatus.F):

    # Report by language unit iterating on lu_unit_set.
    words = []
    family = ''
    language = ''
    concept = ''

    # Sort by language family, language, and concept.
    table_ = sorted(table, key=lambda table_row:
                    (table_row[0], table_row[1], table_row[8]))
    for row in table_:
        cogid_ = row[4]

        # If only reporting identified cognates, then skip if no cogid.
        if require_cogid and cogid_ == 0: continue

        # Any cogid > 0 interpreted as borrowed otherwise inherited.
        # For donor candidate focus, need to insert test for donor language.
        # Consider if donor not included in cogid.
        # *** #

        pred = 0 if cogid_ == 0 else 1
        loan = row[5]
        status_ = util.assess_pred(pred, loan)

        # Check whether status needs to be reported.
        if not util.report_assessment(status, status_): continue

        # Do we start a new language family?
        if family != row[0]: words.append([])
        family_ = row[0] if family != row[0] else ''
        family = row[0]
        # Do we start a new language?
        if language != row[1]: words.append([])
        language_ = row[1] if language != row[1] else ''
        language = row[1]
        # Do we start a new concept?
        concept_ = row[7] if concept != row[7] else ''
        concept = row[7]

        borrowed_ = True if loan else False
        tokens_ = row[6]
        donor_language = row[9] if len(row) > 9 else ''
        donor_value = row[10] if len(row) > 10 else ''

        status_name = status_.name

        # Use concept name.
        concept_name_ = '' if not concept_ else row[8]  # concept_name

        # Add donor forms to reporting.
        # Donor forms are indexed by cogid so only available for cogid>0.
        if cogid_ != 0:
            donors = donor_forms[cogid_]
            # candidate donors indexed by cogid.
            marked = '*'
        else:
            donors = unmarked_donor_forms[row[7]]
            # candidate donors indexed by concept_id
            marked = ' '

        if len(donors.items()) == 0:
            words.append([family_, language_,
                          f'{concept_name_:.20}',
                          f'{tokens_:.40}',
                          donor_language,
                          f'{donor_value:20}',
                          cogid_, borrowed_, status_name])
        else:
            for candidate, candidate_tokens in donors.items():
                words.append([family_, language_,
                              f'{concept_name_:.20}',
                              f'{tokens_:.40}',
                              donor_language,
                              f'{donor_value:.20}',
                              cogid_, borrowed_,  status_name,
                              candidate, marked,
                              '{:.50}'.format(f'{candidate_tokens}')])
                family_ = ''
                language_ = ''
                concept_name_ = ''
                cogid_ = ''
                borrowed_ = ''
                tokens_ = ''
                status_name = ''
                donor_language = ''
                donor_value = ''

    return words


def report_words_table(words, 
                       threshold=None,
                       output=None, 
                       series='',
                       first_time=True):

    filename = f"cluster-{threshold:0.2f}-{series}{'-' if series else ''}words-status"
    file_path = Path(output).joinpath(filename).as_posix()
    header = ['Family', 'Language', 'Concept',  'Tokens',
              'Donor Language', 'Donor Value', 'Cog_id',
              'Borrowed', 'Status', 'Donor Candidate', 'Mark',
              'Candidate Tokens']
    words_table = tabulate(words, headers=header, tablefmt="pip")
    with open(file_path + '.txt', 'w' if first_time else 'a') as f:
        print(f"Threshold: {threshold:0.3f}.", file=f)
        print(words_table, file=f)


def get_metrics_by_language_unit(table, lu_units, lu_idx):
    # Consider all together.
    pred = [0 if row[4] == 0 else 1 for row in table]
    loan = [1 if row[5] else 0 for row in table]
    # **** insert test for donor language.
    # NOT here; we don't have the details here.
    # *** #

    qa = util.prf(pred, loan)

    # Report by language unit iterating on lu_unit_set.
    # Wasteful since not taking advantage of sorted table, but it works.
    metrics = []
    for lu in lu_units:
        pred = [0 if row[4] == 0 else 1 for row in table if row[lu_idx] == lu]
        loan = [1 if row[5] else 0 for row in table if row[lu_idx] == lu]
        q = util.prf(pred, loan)
        metrics.append([lu] + q)
    metrics.append(['Total'] + qa)
    return metrics


def build_donor_forms_dict(cogids_table, donors):
    # Construct candidate donor forms for marked candidate donor
    # for donor family and language indexed by cogid.
    # Only map for the same cognate table.
    # Save form as string of tokens.
    # Also construct unmarked candidate donors indexed by concept.
    # From cogids_table:
    #     family_, language_, local_, global_, global_gt1_,
    #     loan_, tokens_, concepts_, concept_names_
    # Use: family_, language_, global_gt1_, tokens_
    # Row: 0,       1,         4,           6
    # For unmarked use: family_, language_, concepts_, tokens_
    # Row:              0,       1,         7,         6

    donor_stuff = defaultdict(lambda: defaultdict(list))
    unmarked_donor_stuff = defaultdict(lambda: defaultdict(list))

    for row in cogids_table:
        if row[1] in donors:
            # Any cogid > 0 interpreted as borrowed otherwise inherited.
            # For donor candidate focus, need to insert test for donor language.
            # Consider if donor not included in cogid.
            # *** #
            if row[4] > 0:
                # tokens = ' '.join(row[6]) if isinstance(row[6], list) else row[6]
                donor_stuff[row[4]][row[1]].append(row[6])
            else:  # Unmarked since no cogid>0
                unmarked_donor_stuff[row[7]][row[1]].append(row[6])

    return donor_stuff, unmarked_donor_stuff


def report_metrics_by_family(cogids_table,
                             family=None,
                             by_fam=False,
                             donors=None,
                             require_cogid=False,
                             report_status=None,
                             threshold=None,
                             output=None,
                             series=''):
    # Construct dictionary of candidate donor forms
    # of possible borrowed words for words report.
    donor_forms, unmarked_donor_forms = build_donor_forms_dict(
        cogids_table, donors=donors)

    table, lu_units, lu_idx = get_language_unit_table(
        cogids_table, family=family, byfam=by_fam, donors=donors)
    metrics = get_metrics_by_language_unit(table, lu_units=lu_units, lu_idx=lu_idx)
    util.report_metrics_table(metrics, by_fam=by_fam, threshold=threshold)

    words = get_words_results(table,
                              donor_forms=donor_forms,
                              unmarked_donor_forms=unmarked_donor_forms,
                              require_cogid=require_cogid,
                              status=report_status)
    report_words_table(words, 
                       threshold=threshold, 
                       output=output, 
                       series=series)


def register(parser):
    parser.add_argument(
        "--store",
        type=str,
        default='store',
        help='Directory from which to load analysis wordlist.',
    )
    parser.add_argument(
        "--infile",
        type=str,
        default="cluster-analysis"
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Family name to report or None if report over all families."
    )
    parser.add_argument(
        "--byfam",
        action="store_true",
        help='Report by family not individual language.'
    )
    parser.add_argument(
        "--cogid",
        action="store_true",
        help='Report only concepts and words with cross family cognate id.'
    )
    parser.add_argument(
        "--status",
        type=str,
        default='ntn',
        choices=[f"{s.name.lower()}" for s in util.PredStatus],
        # choices=["tn", "tp", "fp", "fn", "f", "t", "ntn", "all"],
        help='Code for reporting words for status.',
    )
    parser.add_argument(  # Added only to create donors list for exclusion in metrics.
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


def run(args):
    table, parameters = get_table(args.store, args.infile)
    thresholds = util.get_thresholds(parameters[-1])
    for idx, threshold in enumerate(thresholds):
        report_basic(table, index=idx)
        cogids_table = get_cogids_table_for(table, index=idx, donors=args.donor, 
                                            # Donors list used to count only wrt donors.
                                            # any_donor_language turns off donor only metrics.
                                            any_loan=args.anyloan)
        report_cogids_by_family(cogids_table,
                                family=args.family,
                                byfam=args.byfam,
                                threshold=threshold)
        report_metrics_by_family(cogids_table,
                                 family=args.family,
                                 by_fam=args.byfam,
                                 donors=args.donor,
                                 require_cogid=args.cogid,
                                 report_status=util.PredStatus[args.status.upper()],
                                 threshold=threshold,
                                 output=args.output,
                                 series=args.series)


def get_total_run_result(store, infile, family, donors, index=0):
    # Application interface to perform run based on invocation by another application.
    # Purpose is to automate experimentation.
    table, parameters = get_table(store, infile)
    cogids_table = get_cogids_table_for(table, index=index, donors=donors)
    table_, lu_units, lu_idx = get_language_unit_table(
        cogids_table, family=family, byfam=False, donors=donors)
    metrics = get_metrics_by_language_unit(table_, lu_units=lu_units, lu_idx=lu_idx)
    q = metrics[-1]
    # print(q)
    result = q[1:]
    result = [round(num, 3) for num in result]
    return result


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
