"""
    Functions specific to Pairwise analysis, evaluation, and reporting.

    John E. Miller, Apr 18, 2022
"""

import copy
from tabulate import tabulate
from lingpy import Pairwise
import sabor.accessdb as adb


def calculate_pairwise_distance(wl, entry1, entry2):
    pair = Pairwise(wl[entry1, 'tokens'], wl[entry2, 'tokens'])
    pair.align(
        distance=True, model='sca', mode='overlap', gop=-1.0,
        scale=0.5, factor=0.3)
    _, _, distance = pair.alignments[0]
    return distance


def donor_entry_is_cognate(
        wl, donor_entry, candidate_donor_entry, within_threshold):
    distance = calculate_pairwise_distance(
        wl, donor_entry, candidate_donor_entry)
    return distance <= within_threshold


def similar_cognate_id(wl, donor_assignments, candidate, within_threshold):
    # Look through donor assignments to see if distance between donors
    # less than within_threshold.
    # Add cross_id and return for this candidate; otherwise return 0.

    candidate_donor_entry = candidate['donor_entry']
    min_distance, min_cross_id = float('inf'), 0
    for donor_entry, cid in donor_assignments.items():
        distance = calculate_pairwise_distance(
            wl, donor_entry, candidate_donor_entry)

        if distance < min_distance:
            min_distance, min_cross_id = distance, cid

    if min_distance <= within_threshold:  # Use existing cross_id
        donor_assignments[candidate['donor_entry']] = min_cross_id
        return min_cross_id

    return 0  # No similar cognate found.


def accept_candidate(wl, cross_id, refs, candidate):
    wl[candidate['entry'], refs['cross_id']] = cross_id
    wl[candidate['entry'], refs['donor_entry']] = candidate['donor_entry']
    wl[candidate['entry'], refs['distance']] = candidate['distance']


def make_cognate_wordlist(wl, bor_book, thresholds, within_threshold):
    # Construct new wordlist similar to 'analyzecluster' wordlist
    # with respect to cross family cognate identifiers for donors.
    # Cognate identifiers are based on similarity to donor words.
    # Additionally each non-donor word entry may reference a most
    # similar donor entry with distance less than threshold.

    # Bor_book nesting is concept, language, entry.
    # Wordlist (wl) will be modified to include additional columns
    # for each threshold of cross_id_{}, donor_entry_{}, distance_{}.

    # Testing ...

    wl = copy.deepcopy(wl)
    # print(f"Header {wl.header}")
    for i, threshold in enumerate(thresholds):
        refs = {'cross_id': "cross_id_{}".format(i),
                'donor_entry': "donor_entry_{}".format(i),
                'distance': "distance_{}".format(i)}

        wl.add_entries(refs['cross_id'], 'tokens', lambda x: 0)
        wl.add_entries(refs['donor_entry'], 'tokens', lambda x: 0)
        wl.add_entries(refs['distance'], 'tokens', lambda x: 0.0)

        cross_id = 0  # Global identifier for this threshold.
        for concept, languages in bor_book.items():
            # donors already excluded from bor_book.
            # determine cross_id, donor_entry, distance for language entries.
            candidates = []
            for language, entries in languages.items():
                for entry, donor_hits in entries.items():
                    for donor, distance, donor_entry in donor_hits:
                        if distance <= threshold:
                            candidates.append(
                                {'entry': entry,
                                 'donor': donor,
                                 'distance': distance,
                                 'donor_entry': donor_entry})

            # List of candidate language recipient, donor pairs ordered by distance.
            candidates = sorted(candidates, key=lambda can: can['distance'])
            donor_assignments = dict()

            for candidate in candidates:  # Subsequent candidates.
                if donor_entry := wl[candidate['entry'], refs['donor_entry']]:
                    # Already assigned with shorter distance.
                    # Try to add candidate donor entry as cognate.
                    candidate_donor_entry = candidate['donor_entry']
                    if donor_entry_is_cognate(wl, donor_entry,
                            candidate_donor_entry, within_threshold):
                        donor_assignments[candidate_donor_entry] = \
                            donor_assignments[donor_entry]
                elif candidate['donor_entry'] in donor_assignments:
                    cid = donor_assignments[candidate['donor_entry']]
                    accept_candidate(wl, cid, refs, candidate)
                elif cid := similar_cognate_id(
                        wl, donor_assignments, candidate, within_threshold):
                    accept_candidate(wl, cid, refs, candidate)
                else:  # New cross_id needed.
                    cross_id += 1
                    accept_candidate(wl, cross_id, refs, candidate)
                    donor_assignments[candidate['donor_entry']] = cross_id

            # Post cross_ids to donor entries.
            for entry, cross_id in donor_assignments.items():
                wl[entry, refs['cross_id']] = cross_id

    return wl


def count_donor_cognates(wl, bor_book, threshold, donors):
    # Report on cognates with donor languages for each threshold.
    # via simple diagnostic indication of results of analysis.

    families = adb.get_language_family(wl)
    print(f"Report on cognates for threshold {threshold}")

    proportions = []
    for language, concepts in bor_book.items():
        family = families[language]
        prop = {donor: [] for donor in donors}
        prop['Combined'] = []
        concept_count = 0

        for concept, entries in concepts.items():
            tmp_prop = {donor: 0 for donor in donors}
            for entry, donor_hits in entries.items():
                for donor, dist, donor_entry in donor_hits:
                    # Possible donor cognate with entry.
                    if dist <= threshold: tmp_prop[donor] += 1

            if entries:  # entries must not be empty.
                # Accumulate proportions.
                max_score = 0  # For use with combined donors category.
                for donor, score in tmp_prop.items():
                    max_score = max(max_score, score)
                    prop[donor] += [score/len(entries)]
                    # Fraction of entries cognate with each donor.
                prop['Combined'] += [max_score/len(entries)]
                # Maximum fraction of entries over donors that are cognate.
                concept_count += 1

        proportions += [[
            family, language, concept_count] +
            [sum(prop[donor]) for donor in donors] +
            [sum(prop['Combined'])] +
            # Unweighted average of entries over concepts for each language.
            [sum(prop[donor])/concept_count for donor in donors] +
            [sum(prop['Combined'])/concept_count]
        ]

    return proportions


def print_donor_proportions(proportions, threshold, donors):

    print(f"\nPairwise alignment with threshold {threshold:0.3f}.")
    headers = ["Family", "Language", "Concepts"] + \
              [donor for donor in donors] + ['Combined'] + \
              [donor + 'P' for donor in donors] + ['CombinedP']

    # Calculate total borrowed.
    concepts_count = 0
    combined_count = 0
    donor_counts = [0]*len(donors)
    for row in proportions:
        concepts_count += row[2]
        for d in range(len(donors)):
            donor_counts[d] += row[d+3]
        combined_count += row[len(donors)+3]
    donor_proportions = [count/concepts_count for count in donor_counts]
    combined_proportion = combined_count/concepts_count
    total_row = (['Total', '', concepts_count] +
                 donor_counts + [combined_count] +
                 donor_proportions + [combined_proportion])

    proportions = sorted(proportions, key=lambda x: (x[0], x[1]))
    proportions.append(total_row)

    print(tabulate(proportions, headers=headers, tablefmt="pip", floatfmt=".2f"))
