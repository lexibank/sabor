"""
    Utility functions not directly related to functions
    analyze, retrieve, evaluate, report, or experiment.

    John E. Miller, Apr 18, 2022
"""

from collections import defaultdict


# Swapping Hierarchy in Nested Dictionaries
# Using defaultdict() + loop
# Algorithm from: https://www.geeksforgeeks.org/
# python-swapping-hierarchy-in-nested-dictionaries/
def swap_dict_top_levels(dict_in):
    dict_out = defaultdict(dict)
    for key, val in dict_in.items():
        for key_in, val_in in val.items():
            dict_out[key_in][key] = val_in
    return dict(dict_out)


if __name__ == "__main__":
    # initializing dictionary
    test_dict = {'Gfg': {'a': [1, 3], 'b': [3, 6], 'c': [6, 7, 8]},
                 'Best': {'a': [7, 9], 'b': [5, 3, 2], 'd': [0, 1, 0]}}
    # printing original dictionary
    print("The original dictionary : " + str(test_dict))
    test_out = swap_dict_top_levels(test_dict)
    print("The rearranged dictionary : " + str(test_out))
