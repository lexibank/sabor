"""
    Diagnostics reporting functions.

    John E. Miller, Apr 15, 2022
"""

from pathlib import Path
from collections import defaultdict
from tabulate import tabulate
import math
from enum import Enum


# ===========================================
# Detection status functions for diagnostics.
# ===========================================
class PredStatus(Enum):
    TN = 1
    TP = 2
    FP = 3
    FN = 4
    F = 5
    T = 6
    NTN = 7
    ALL = 0


#  Functions adapted from pybor for assessment.
def assess_pred(pred, gold, any_pred=None):
    """
    Test 0, 1 prediction versus 0, 1 truth
    """
    if any_pred and gold == 1:
        if pred == 0: return PredStatus.TN

    if pred == gold:
        if gold == 0: return PredStatus.TN
        elif gold == 1: return PredStatus.TP
    else:
        if gold == 0: return PredStatus.FP
        elif gold == 1: return PredStatus.FN
    raise ValueError(f"Pred {pred}, gold {gold}.")


def report_assessment(status, result):
    return (status == PredStatus.ALL or
            (status == PredStatus.F and
             (result == PredStatus.FN or result == PredStatus.FP)) or
            (status == PredStatus.T and
             (result == PredStatus.TN or result == PredStatus.TP)) or
            (status == PredStatus.NTN and result != PredStatus.TN) or
            result == status)


# ==========================================================
# Report detail diagnostics selected from individual entries
# ==========================================================
def build_donor_forms_dict(ids_table, donors):
    # Construct candidate donor forms for marked candidate donor
    # for donor family and language indexed by cross_fam_id.
    # Only map for the same cognate table.
    # Save form as string of tokens.
    # Construct unmarked candidate donors indexed by concept.

    donor_stuff = defaultdict(lambda: defaultdict(list))
    unmarked_donor_stuff = defaultdict(lambda: defaultdict(list))

    for row in ids_table:
        if row["DOCULECT"] in donors:
            if row["CROSS_FAMILY_ID"] > 0:
                donor_stuff[row["CROSS_FAMILY_ID"]][row["DOCULECT"]].\
                    append(row["TOKENS"])
            else:  # Unmarked since no cross family id
                unmarked_donor_stuff[row["CONCEPT"]][row["DOCULECT"]].\
                    append(row["TOKENS"])

    return donor_stuff, unmarked_donor_stuff


def get_diagnostics(ids_table, donor_forms, unmarked_donor_forms,
                    xfid_only=False, status=PredStatus.F):

    diagnostics = []
    family = ''
    language = ''
    concept = ''

    # Sort by language family, language, and concept.
    table_ = sorted(ids_table, key=lambda table_row:
                    (table_row["FAMILY"], table_row["DOCULECT"],
                     table_row["CONCEPT_NAME"]))
    for row in table_:
        cross_fam_id = row["CROSS_FAMILY_ID"]

        # If only reporting identified cognates, then skip if no xfid.
        if xfid_only and cross_fam_id == 0: continue

        pred = 0 if cross_fam_id == 0 else 1
        loan = row["BORROWED"]
        status_ = assess_pred(pred, loan)

        # Check whether status needs to be reported.
        if not report_assessment(status, status_): continue

        # Do we start a new language family?
        if family != row["FAMILY"]: diagnostics.append([])
        family_ = row["FAMILY"] if family != row["FAMILY"] else ''
        family = row["FAMILY"]
        # Do we start a new language?
        if language != row["DOCULECT"]: diagnostics.append([])
        language_ = row["DOCULECT"] if language != row["DOCULECT"] else ''
        language = row["DOCULECT"]
        # Do we start a new concept?
        concept_ = row["CONCEPT"] if concept != row["CONCEPT"] else ''
        concept = row["CONCEPT"]

        borrowed_ = True if loan else False
        tokens_ = row["TOKENS"]
        donor_language = row["DONOR_LANGUAGE"] if "DONOR_LANGUAGE" in row else ''
        donor_value = row["DONOR_VALUE"] if "DONOR_VALUE" in row else ''

        status_name = status_.name

        # Use concept name.
        concept_name_ = '' if not concept_ else row["CONCEPT_NAME"]  # concept_name

        # Add donor forms to reporting.
        # Donor forms are indexed by cogid so only available for cogid>0.
        if cross_fam_id != 0:
            donors = donor_forms[cross_fam_id]
            # candidate donors indexed by cogid.
            marked = '*'
        else:
            donors = unmarked_donor_forms[row["CONCEPT"]]
            # candidate donors indexed by concept_id
            marked = ' '

        if len(donors.items()) == 0:
            diagnostics.append([family_, language_,
                                f'{concept_name_:.20}',
                                f'{tokens_:.40}',
                                donor_language,
                                f'{donor_value:20}',
                                cross_fam_id, borrowed_, status_name])
        else:
            for candidate, candidate_tokens in donors.items():
                diagnostics.append([family_, language_,
                                    f'{concept_name_:.20}',
                                    f'{tokens_:.40}',
                                    donor_language,
                                    f'{donor_value:.20}',
                                    cross_fam_id, borrowed_,  status_name,
                                    candidate, marked,
                                    '{:.50}'.format(f'{candidate_tokens}')])
                family_ = ''
                language_ = ''
                concept_name_ = ''
                cross_fam_id = ''
                borrowed_ = ''
                tokens_ = ''
                status_name = ''
                donor_language = ''
                donor_value = ''

    return diagnostics


def report_diagnostics(diagnostics,
                       threshold=float('Nan'),
                       output=None,
                       series='',
                       module=''):

    threshold_str = "{:0.3f}-".format(threshold) \
        if not math.isnan(threshold) else ''
    filename = f"{module}-{threshold_str}{series}{'-' if series else ''}word-diagnostics"
    file_path = Path(output).joinpath(filename).as_posix()
    header = ['Family', 'Language', 'Concept',  'Tokens',
              'Donor Language', 'Donor Value', 'Bor_id',
              'Borrowed', 'Status', 'Donor Candidate', 'Mark',
              'Candidate Tokens']
    words_table = tabulate(diagnostics, headers=header, tablefmt="simple")
    with open(file_path + '.txt', 'w') as f:
        print(f"Threshold: {threshold:0.3f}.", file=f)
        print(words_table, file=f)


def report_detail_diagnostics(
        ids_table,
        donor_forms=None,
        unmarked_donor_forms=None,
        donors=None,
        status=None,
        threshold=float('NaN'),
        output=None,
        series='',
        module=''):

    if donors:
        ids_table = [row for row in ids_table if row["DOCULECT"] not in donors]

    diagnostics = get_diagnostics(
        ids_table,
        donor_forms=donor_forms,
        unmarked_donor_forms=unmarked_donor_forms,
        status=status)

    report_diagnostics(
        diagnostics,
        threshold=threshold,
        output=output,
        series=series,
        module=module)
