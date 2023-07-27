"""
Describe individual languages.
"""

import pathlib
import pycldf
from lexibank_sabor import get_our_wordlist
from collections import defaultdict
from tabulate import tabulate


def run(args):
    wl = get_our_wordlist()
    args.log.info("Describe language varieties from SaBor database.")

    # print(wl.columns)

    tokens = defaultdict(list)
    concepts = defaultdict(list)

    for idx in wl:
        lang = wl[idx, 'doculect']
        tokens[lang] += wl[idx, 'tokens']
        concepts[lang].append(wl[idx, 'concept'])

    db_dir = pathlib.Path(__file__).parent.parent
    cldf = pycldf.Dataset.from_metadata(db_dir / "cldf" / "cldf-metadata.json")
    langs = cldf.objects("LanguageTable")

    languages = dict()

    for entry in langs:
        data = entry.data
        # print(data)

        languages[data['ID']] = \
            {'Name': data['Name'],
             'Latitude': round(float(data['Latitude']), 2),
             'Longitude': round(float(data['Longitude']), 2),
             'Spanish_Borrowings': round(float(data['Spanish_Borrowings']), 2)
                if data['Spanish_Borrowings'] else None,
             'Borrowing_Class': data['Borrowing_Class'],
             'Tokens': len(tokens[data['ID']]),
             'Segments': len(set(tokens[data['ID']])),
             'Lexemes': len(concepts[data['ID']]),
             'Concepts': len(set(concepts[data['ID']]))
             }

    # print(languages)
    print(tabulate(languages.values(),
                   headers="keys", tablefmt="pip", floatfmt=".2f"))

