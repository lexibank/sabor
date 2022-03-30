"""
    Report on pairwise alignments with Spanish and Portuguese languages.

    Johann-Mattis List, Aug 22, 2021
    John Edward Miller, Oct 6, 2021
"""
import sys
from pathlib import Path
import argparse

from distutils.util import strtobool
from lingpy import *
# from lingpy.sequence.sound_classes import token2class

from lexibank_sabor import Dataset as sabor
from itertools import product
from tabulate import tabulate
from pylexibank import progressbar
import saborcommands.util as util


def construct_alignments(wl, model, mode, gop, donors, pbs=False, min_len=1, config=None):
    # Construct borrowing-bookkeeping structure.
    # Config has positional based scorer weights.
    bb = {doculect: {} for doculect in wl.cols
          if not any((lambda x, y: x in y)(donor, doculect) for donor in donors)}

    sum_dist = 0.0
    cnt_dist = 0
    for concept in progressbar(wl.rows, desc="pairwise SCA"):
        # idxs are all entries for given concept.
        # Entries correspond to doculect form and related data.
        idxs = wl.get_dict(row=concept)
        # Populate donors indices for concept.  This collects the entries for this donor and concept.
        donors_concepts = {donor: idxs.get(donor, []) for donor in donors}
        for doculect in idxs:
            # Skip the doculect if a donor langauge.
            if any((lambda x, y: x in y)(donor, doculect) for donor in donors): continue
            # if "Spanish" not in doc and "Portuguese" not in doc:
            # For this doculect and concept, construct a dictionary of entries.
            # Below we add list of aligned donor words corresponding to this entry.
            bb[doculect][concept] = {idx: [] for idx in idxs[doculect]}
            for donor in donors:
                # All combinations of donor and doculect entries for this concept.
                # Often just 1 entry each, but can be multiple entries for concept.
                for idxA, idxB in product(donors_concepts[donor], idxs[doculect]):
                    # Combination of donor and doculect entries.
                    # idxA is from donor, and idxB from doculect.
                    wordA, wordB = wl[idxA, "tokens"], wl[idxB, "tokens"]
                    pair = Pairwise(wordA, wordB)
                    # ** Add scale and factor to align.
                    pair.align(
                        distance=True, model=model, mode=mode, gop=gop,
                        scale=config["scale"] if config and config.get("scale") else 0.5,
                        factor=config["factor"] if config and config.get("factor") else 0.3)
                    almA, almB, dist = pair.alignments[0]  # [0] indexes alignment of only pair

                    pbs_dist = 1.0
                    if pbs:
                        pbs_dist = util.relative_position_based_scoring(
                            almA, almB, gop=gop, model=model,
                            weights=config["weights"] if config and config.get("weights") else None,
                            min_len=config["min_len"] if config and config.get("min_len") else min_len,
                            wt_fn=config["wt_fn"] if config and config.get("wt_fn") else 2)

                    # Save candidate donor word reference to store.  Save loanword status to store.
                    # print(f"Dist={dist}, pbs_dist={pbs_dist}")
                    dist = dist if pbs_dist == 1.0 else pbs_dist
                    # WordA is potential loan word from candidate donor language
                    # Not the documented donor value!
                    # Alignments indexed by doculect, concept, and doculect entry.
                    # Value is list of candidate donor words as:
                    # (candidate donor, distance from doculect tokens, donor tokens).
                    bb[doculect][concept][idxB] += [(donor, dist, wordA)]
                    sum_dist += dist
                    cnt_dist += 1

    print(f"Ave dist = {sum_dist/cnt_dist:0.2f}")
    return bb


def screen_word_hits(tmp_words, threshold):
    # Organize according to requirements of discrimination between donors.
    # Get index of min distance word.
    min_idx = min(enumerate(tmp_words), key=lambda x: x[1][1])[0]
    min_dist = tmp_words[min_idx][1]
    min_idx = min_idx if min_dist < threshold else None
    tmp_words_ = []
    for idx, row in enumerate(tmp_words):
        if min_idx is None:
            tmp_words_ += [row + ['']]
        elif idx == min_idx:
            tmp_words_ += [row + ['*']]
        elif row[1] <= threshold:
            tmp_words_ += [row + ['-']]
        else:
            tmp_words_ += [row + ['']]
    return tmp_words_


def order_words_table(table, status=util.PredStatus.NTN):
    # Remove redundant use of family, language, concept, and word.
    # Sort table by family.  Stable sort so rest of order should be OK.
    # Change unchanged fields to blank after sort for better presentation.
    table = sorted(table, key=lambda table_row: (table_row[0], table_row[1], table_row[2]))
    ordered_words = []
    family = ''
    language = ''
    concept = ''
    word = ''
    for row in table:
        status_ = row[9]
        if not util.report_assessment(status, status_): continue

        family_ = row[0] if family != row[0] else ''
        family = row[0]
        language_ = row[1] if language != row[1] else ''
        language = row[1]
        concept_ = row[2] if concept != row[2] else ''
        concept = row[2]
        word_ = row[3] if word != row[3] else ''
        word = row[3]
        borrowed_ = 'True' if row[4] and word_ else 'False' if not row[4] and word_ else ''

        donor_candidate = row[5]
        distance = row[6]
        marker = row[8]
        candidate_tokens = row[7]
        # Donor tied to form. If same form, we don't repeat donor info.
        donor_language = row[10] if word_ else ''
        donor_value = row[11] if word_ else ''

        ordered_words.append([family_, language_, concept_, word_,
                              donor_language, donor_value,
                              borrowed_, status_.name, donor_candidate,
                              distance, marker, candidate_tokens])

    return ordered_words


def report_words_table(words, threshold, output, series):

    filename = f"pairwise-{threshold:0.3f}-{series}{'-' if series else ''}words-status"
    file_path = Path(output).joinpath(filename).as_posix()
    header = ['Family', 'Language', 'Concept',  'Status', 'Tokens']
    words_table = tabulate(words, headers=header, tablefmt="pip")
    with open(file_path + '.txt', 'w') as f:
        print(f"Threshold: {threshold:0.3f}.", file=f)
        print(words_table, file=f)


def report_pairwise_distance(words_table, threshold, output, series):
    # Report out.
    headers = ["Family", "Language", "Concept", "Tokens",
               "Donor Language", "Donor Value", "Borrowed", "Status",
               "Donor Candidate", "Distance", "Mark", "Candidate Tokens"]
    words_table = tabulate(words_table, headers=headers, tablefmt="simple")
    filename = f"pairwise-{threshold:0.3f}-{series}{'-' if series else ''}words-distance.txt"
    filepath = Path(output).joinpath(filename).as_posix()
    with open(filepath, 'w') as f:
        print(f"Threshold: {threshold:0.3f}.", file=f)
        print(words_table, file=f)
        print(file=f)


def report_donor_proportions(proportions, threshold, donors):

    print(f"\nPairwise alignment with threshold {threshold:0.3f}.")
    headers = ["Family", "Language", "Concepts"] + [
        donor for donor in donors] + ['Combined'] + [
                  donor + 'P' for donor in donors] + ['CombinedP']

    # Calculate total borrowed.
    concepts_count = 0
    combined_count = 0
    donor_counts = [0]*len(donors)
    for row in proportions:
        concepts_count += row[2]
        for d in range(len(donors)):
            donor_counts[d] += row[3+d]
        combined_count += row[len(donors)+3]
    donor_proportions = [count/concepts_count for count in donor_counts]
    combined_proportion = combined_count/concepts_count
    total_row = (['Total', '', concepts_count] +
                 donor_counts + [combined_count] +
                 donor_proportions + [combined_proportion])

    proportions = sorted(proportions, key=lambda x: (x[0], x[1]))
    proportions.append(total_row)

    print(tabulate(proportions, headers=headers, tablefmt="pip", floatfmt=".2f"))


def get_overall_detection(all_words):
    pred = [1 if row[4] else 0 for row in all_words]
    loan = [1 if row[5] else 0 for row in all_words]
    return util.prf(pred, loan)


def report_pairwise_detection(all_words, threshold):
    # Report detection metrics by language
    # language, family, concept, word, pred, loan in all_words:
    # pred = [1 if row[4] else 0 for row in all_words]
    # loan = [1 if row[5] else 0 for row in all_words]
    # q = util.prf(pred, loan)
    q = get_overall_detection(all_words)

    def calculate_metrics_table(table, lu_units, lu_idx):
        metrics_ = []
        for lu in lu_units:
            pred_ = [1 if row[4] else 0 for row in table if row[lu_idx] == lu]
            loan_ = [1 if row[5] else 0 for row in table if row[lu_idx] == lu]
            metrics_.append([lu] + util.prf(pred_, loan_))
        return metrics_

    languages = sorted(set(row[1] for row in all_words))
    metrics = calculate_metrics_table(all_words, lu_units=languages, lu_idx=1)
    metrics.append(['Total'] + q)
    util.report_metrics_table(metrics, byfam=False, threshold=threshold)

    families = sorted(set(row[0] for row in all_words))
    metrics = calculate_metrics_table(all_words, lu_units=families, lu_idx=0)
    metrics.append(['Total'] + q)
    util.report_metrics_table(metrics, byfam=True, threshold=threshold)


def get_words_results(table, status=util.PredStatus.F):
    words = []
    family = ''
    language = ''
    concept = ''
    table_ = sorted(table, key=lambda table_row: table_row[2])
    table_ = sorted(table_, key=lambda table_row: table_row[1])
    table_ = sorted(table_, key=lambda table_row: table_row[0])
    for row in table_:
        # pred == True if global_gt1
        pred = int(row[4])
        loan = int(row[5])
        status_ = util.assess_pred(pred, loan)
        if util.report_assessment(status, status_):
            family_ = row[0] if family != row[0] else ''
            family = row[0]
            language_ = row[1] if language != row[1] else ''
            language = row[1]
            concept_ = row[2] if concept != row[2] else ''
            concept = row[2]
            words.append([family_, language_, concept_, status_.name, row[3], ])
            # family, language, concept, prediction_result, tokens,
    return words


def detect_borrowing(wl, bb,
                     families,
                     threshold,
                     donors,
                     report_limit=None,
                     any_donor_language=False):

    proportions = []  # Used for report of donor proportions.
    words = []  # Used for subsequent reporting of distances below threshold.
    all_words = []  # Used for subsequent calculation of detection metrics.
    for language, concepts in bb.items():
        family = families[language]
        # determine proportion of borrowed words
        prop = {donor: [] for donor in donors}
        # include combined donors category.
        prop['Combined'] = []
        concept_count = 0
        no_concept_id_count = 0
        # idx is index of word in target language
        for concept, idxs in concepts.items():

            tmp_prop = {donor: 0 for donor in donors}
            # Index to single word entry in target language.
            # For some concepts, there is no given donor word concept.
            for idx, hits in idxs.items():
                word = wl[idx, 'tokens']
                loan = wl[idx, 'loan'] if wl[idx, 'loan'] else wl[idx, 'borrowed']
                loan = loan if not isinstance(loan, str) else strtobool(loan)

                # Focus on borrowings from donor languages only.
                donor_language = wl[idx, 'donor_language']
                donor_value = wl[idx, 'donor_value']
                # Loan set True if donor language is one of candidate donors.
                # May need to improve this.  e.g., Spanish (Mexican) does not match!
                # Indicator to apply to only donor language as default.
                if loan and not any_donor_language:
                    if not any(donor.startswith(wl[idx, 'donor_language'])
                               for donor in donors): loan = False

                anyPred = False  # Any distance < threshold will qualify

                tmp_words = []
                for donor, dist, donor_word in hits:
                    if dist <= threshold or (report_limit and dist <= report_limit):
                        tmp_words += [[donor, dist, donor_word]]
                    if dist <= threshold:  # Only need 1 donor word < threshold.
                        anyPred = True

                # Add word to all_words for words status report.
                # Words are target language words, not possible donor words.            
                concept_name = wl[idx, 'concept_name'] if 'concept_name' in wl.columns else concept
                concept_name = concept_name.lower()

                all_words.append([families[language], language,
                                  concept_name, word, anyPred, loan])
                # print(f"All words {language}, {concept_name}, {word}")

                if not tmp_words: continue  # Nothing to add to distance report
                # Add marker of minimum '*' or near minimum '-' < threshold.
                tmp_words = screen_word_hits(tmp_words, threshold)
                for row in tmp_words:
                    # Count hits for distance < threshold.
                    if row[1] < threshold:
                        tmp_prop[row[0]] += 1
                    # Report words < threshold, or words < report_limit.
                    if row[1] < threshold or (report_limit and row[1] <= report_limit):
                        pred_ = 1 if row[3] in ['*', '-'] else 0
                        # Take into account whether some other form matches.
                        status_ = util.assess_pred(pred_, int(loan), anyPred)
                        distance_ = row[1]
                        donor_candidate_ = row[0]
                        candidate_tokens_ = row[2]
                        marker_ = row[3]
                        # if status_ in [util.PredStatus.FN, util.PredStatus.FP]:
                        #    candidate_tokens_ = '*' + candidate_tokens_
                        words.append([family, language, concept_name, word, loan,
                                      donor_candidate_, f'{distance_:0.2f}',
                                      candidate_tokens_, marker_, status_,
                                      donor_language, donor_value])

            if idxs:
                # Get max score for use with combined donors category.
                max_score = 0
                for donor, score in tmp_prop.items():
                    max_score = max(max_score, score)
                    prop[donor] += [score/len(idxs)]
                # Add in score for Combined category.
                prop['Combined'] += [max_score/len(idxs)]
                concept_count += 1

        proportions += [[
            families[language],
            language,
            concept_count] +
            [sum(prop[donor]) for donor in donors] +
            [sum(prop['Combined'])] +
            [sum(prop[donor])/concept_count for donor in donors] +
            [sum(prop['Combined'])/concept_count]
        ]

        # print(f"{language} no id concepts count {no_concept_id_count}.")

    return proportions, words, all_words


def report_borrowing(proportions,
                     words,
                     all_words,
                     threshold,
                     report_status,
                     donors,
                     output='output',
                     series=''):

    if series:
        word_distances = order_words_table(table=words, status=report_status)
        report_pairwise_distance(words_table=word_distances,
                                 threshold=threshold,
                                 output=output, series=series)

        word_assessments = get_words_results(table=all_words, status=report_status)
        report_words_table(word_assessments,
                           threshold=threshold,
                           output=output, series=series)

    report_donor_proportions(proportions, threshold, donors)
    report_pairwise_detection(all_words, threshold)


def register(parser):
    parser.add_argument(
        "--language",
        nargs="*",
        type=str,
        default=['all'],
        help="'all' or list of languages"
        #
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sca", "asjp"],
        default="sca",
        help='Sound class model to transform tokens.'
    )
    parser.add_argument(
        "--threshold",
        nargs="*",
        type=float,
        default=[0.4],
        help='Threshold(s) to use with pairwise alignment method.',
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit to use for reporting words and donor candidate distances."
    )
    parser.add_argument(
        "--status",
        type=str,
        default='ntn',
        choices=[e.name.lower() for e in util.PredStatus],
        help="Status mask to use for reporting borrowed word detection status."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overlap",
        choices=["global", "local", "overlap", "dialign"],
        help='Alignment mode.',
    )
    parser.add_argument(
        "--gop",
        type=float,
        default=-1.0,
        help='Gap open penalty.'
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
        "--donor",
        nargs="*",
        type=str,
        default=["Spanish", "Portuguese"],
        help='Donor language(s).',
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
        default="pairwise-donor-words",
        help='Filename prefix for candidate donor words.'
    )
    parser.add_argument(
        "--foreign",
        type=str,
        default=None,
        help="Filename of flat wordlist for analysis from foreign-tables directory."
    )


def run(args):
    filename = args.foreign
    wl = util.get_wordlist(filename)
    
    # Sub-select languages based on languages and donors arguments.
    args.language = util.get_language_all(wl) if args.language[0] == 'all' else args.language
    wl = util.select_languages(wl, languages=args.language, donors=args.donor)

    # Save temp file for testing
    # if filename:  # Test - save wordlist
    #     filepath = Path("store").joinpath(filename+"-TEST").as_posix()
    #    wl.output('tsv', filename=filepath, ignore='all', prettify=False)

    bb = construct_alignments(wl,
                              model=args.model,
                              mode=args.mode,
                              gop=args.gop,
                              donors=args.donor,
                              pbs=args.pbs,
                              min_len=args.min_len)

    # rdr = sabor().cldf_reader()
    # families = {language["ID"]: language["Family"] for language in rdr['LanguageTable']}
    families = util.get_language_family(wl)

    for threshold in args.threshold:
        proportions, words, all_words = detect_borrowing(
            wl, bb,
            families=families,
            threshold=threshold,
            report_limit=args.limit,
            donors=args.donor,
            any_donor_language=False)
        report_borrowing(
            proportions=proportions,
            words=words,
            all_words=all_words,
            threshold=threshold,
            report_status=util.PredStatus[args.status.upper()],
            donors=args.donor,
            output=args.output,
            series=args.series)


def get_total_run_result(languages, donors, config, filename=None):
    # Application interface to perform run based on invocation by another application.
    # Purpose is to automate experimentation.
    # config includes alignment parameters: model, mode, gop, scale, factor;
    # positional based scoring parameters: pbs, slp_factor, slp_model, C, c, V, v, _; and
    # thresholds (list).

    wl = util.get_wordlist(filename)

    # Sub-select languages based on languages and donors arguments.
    languages = util.get_language_all(wl) if languages[0] == 'all' else languages
    wl = util.select_languages(wl, languages=languages, donors=donors)
    # Use config to construct argument invocations to high level functions.
    bb = construct_alignments(
        wl,
        donors=donors,
        model=config["model"],
        mode=config["mode"],
        gop=config["gop"],
        pbs=config["pbs"],
        min_len=config["min_len"],
        config=config)

    # rdr = keypano().cldf_reader()
    # families = {language["ID"]: language["Family"] for language in rdr['LanguageTable']}
    families = util.get_language_family(wl)
    # Can have multiple thresholds in single invocation.
    results = []
    for threshold in config["threshold"]:
        _, _, all_words = detect_borrowing(
            wl, bb,
            families=families,
            threshold=threshold,
            donors=donors,
            any_donor_language=True)

        result = get_overall_detection(all_words)
        result = [round(num, 3) for num in result]
        results.append(result)

    return results


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
