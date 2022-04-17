"""
    Functions implementing position based scorer
    for use with LingPy Pairwise module.
    Functions and argument lists provide the opportunity
    to experiment using an instrumented experimental scripts.

    John E. Miller, Apr 16, 2022
"""

from lingpy.sequence.sound_classes import token2class
from lingpy import *


# ==============================
# Positional based scorers (PBS)
# ==============================
def adjust_dist_for_length(dist, alm, max_len=0, min_len=0):
    # If local mode reduces the alignment size too much,
    # the distance measure may not be reliable.
    # Set distance to 1.0 if alignment cropped to less than minimum length.
    # print(f"Entering dist {dist}, min_len {min_len}, max_len {max_len}, len_alm {len(alm)}")
    if min_len <= 1 or max_len <= 0: return dist  # No test for cropped lenth < mimimum.
    dist = dist if len(alm) >= min_len or len(alm) == max_len else 1.0
    # print(f"Leaving dist {dist}, min_len {min_len}, max_len {max_len}, len_alm {len(alm)}")
    return dist


def position_based_scoring(
        almA, almB, gop=-1, weights=None, scorer=None, model="sca", max_len=0, min_len=1):

    weights = weights or {"C": 1, "c": 0.8, "V": 0.6, "v": 0.4, "_": 0}
    scorer = scorer or rc("model").scorer

    # get consensus string
    consensus, clsA, clsB = [], [], []
    for a, b in zip(almA, almB):
        if a == "-":
            consensus += [b]
            clsA += ["-"]
            clsB += [token2class(b, model)]
        elif b == "-":
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += ["-"]
        else:
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += [token2class(b, model)]
    # get prosodic consensus
    prostring = prosodic_string(consensus, _output="CcV")
    prostringA = prosodic_string([x for x in almA if x != "-"], _output="CcV")
    prostringB = prosodic_string([x for x in almB if x != "-"], _output="CcV")

    # score
    scores, scoresA, scoresB = [], [], []
    for a, b, p in zip(clsA, clsB, prostring):
        if "-" in [a, b]:
            # Doesn't use scorer for -, rather use gop.
            scores += [gop * weights[p]]
        else:
            scores += [scorer[a, b] * weights[p]]

    for a, p in zip([x for x in clsA if x != "-"], prostringA):
        scoresA += [scorer[a, a] * weights[p]]
    for b, p in zip([x for x in clsB if x != "-"], prostringB):
        scoresB += [scorer[b, b] * weights[p]]

    scores_sum = sum(scoresA) + sum(scoresB)
    if scores_sum == 0: return 1.0
    dist = 1-(2 * sum(scores))/scores_sum
    adj_dist = adjust_dist_for_length(dist, almA, max_len=max_len, min_len=min_len)
    # print(f"Dist={dist}, Sum={sum(scores)}, Score: {scores}, "
    #       f"SCA: {clsA}, {clsB}, A {scoresA}, B {scoresB}")
    return adj_dist


def relative_position_based_scoring(
        almA, almB, gop=-2, weights=None, model="sca", min_len=4, wt_fn=2):
    # Starting with previous positional scorer base.

    weights = weights or {"C": 1, "c": 0.8, "V": 0.6, "v": 0.4, "_": 0}
    scorer = rc("model").scorer
    # print('scorer', scorer)

    # get consensus and token alignment class strings
    consensus, clsA, clsB = [], [], []
    for a, b in zip(almA, almB):
        if a == "-":
            consensus += [b]
            clsA += ["-"]
            clsB += [token2class(b, model)]
        elif b == "-":
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += ["-"]
        else:
            consensus += [a]
            clsA += [token2class(a, model)]
            clsB += [token2class(b, model)]
    # print('Classes', clsA, clsB)

    # get prosodic censensus and alignment strings.
    prostring = prosodic_string(consensus, _output="CcV")
    prostringA = prosodic_string([x for x in almA if x != "-"], _output="CcV")
    prostringB = prosodic_string([x for x in almB if x != "-"], _output="CcV")
    # print('* prosodic', prostringA, prostringB)

    # Calculate relative weights.
    # def find_seq_start(str, start):
    left, right, size = get_core_bounds(almA, almB)
    # print(f"left {left}, right {right}, size {size}.")
    if size >= min_len:
        relative_weights = calculate_alm_wts(left, right, len(almA), wt_fn=wt_fn)
    else:
        relative_weights = [1.0]*len(almA)

    # print(f"rel wts: {relative_weights}")

    # calculate score.
    scores, scoresA, scoresB = [], [], []
    for i, (a, b, p) in enumerate(zip(clsA, clsB, prostring)):
        if "-" in [a, b]:
            # Doesn't use weight for -, rather use gop.
            scores += [gop * weights[p] * relative_weights[i]]
        else:
            scores += [scorer[a, b] * weights[p] * relative_weights[i]]

    for i, (a, p) in enumerate(zip([x for x in clsA if x != "-"], prostringA)):
        scoresA += [scorer[a, a] * weights[p] * relative_weights[i]]
    for i, (b, p) in enumerate(zip([x for x in clsB if x != "-"], prostringB)):
        scoresB += [scorer[b, b] * weights[p] * relative_weights[i]]

    scores_sum = sum(scoresA) + sum(scoresB)
    if scores_sum == 0: return 1.0
    dist = 1-(2 * sum(scores))/scores_sum
    # print('scores', scores)
    # print('* scores *', 2*sum(scores), scores_sum)

    return dist


def find_seq_start(alm, dir):
    it = range(0, len(alm), 1) if dir == 1 else range(-1, -len(alm)-1, -1)
    for i in it:  # range(0, len(alm), dir):
        if alm[i] not in ['-', '_', '+']:
            return i
    return None  # Shouldn't happen since words should contain a sequence.


def get_core_bounds(almA, almB):
    left = max(find_seq_start(almA, 1), find_seq_start(almB, 1))
    if not find_seq_start(almA, -1) or not find_seq_start(almB, -1):
        print(f"No start: {almA}, {find_seq_start(almA, -1)}, "
              f"{almB}, {find_seq_start(almB, -1)}")
    right = min(find_seq_start(almA, -1), find_seq_start(almB, -1))
    size = len(almA)+right-left+1
    return left, right, size


def calculate_alm_wts(left, right, ln, wt_fn=2):
    # Calculate wt vector for score.
    mul = 1.0/(2.0 ** (1./wt_fn))
    wts = list([1] * ln)
    wt = 1.0
    for i in range(left-1, -1, -1):
        wt *= mul
        wts[i] = wt
    wt = 1.0
    for i in range(right+1, 0, +1):
        wt *= mul
        wts[i] = wt
    return wts

