"""
    Shared routines for SaBor.

    John E. Miller, Sep 7, 2021
"""
# import tempfile
# from pathlib import Path
# import sys
#
# from lingpy import *
# from lingpy.compare.util import mutual_coverage_check
# from lingpy.compare.sanity import average_coverage
# from lexibank_sabor import Dataset as sabor
#
#
# # ==============
# # Data functions
# # ==============
# def get_language_all(wordlist, donors=None):
#     languages = wordlist.cols
#     if not donors: return languages
#     # Keep only languages that are not donor languages.
#     return [language for language in languages if language not in donors]
#
#
# def select_languages(wordlist=None, languages=None, donors=None):
#     # Use languages and donors to select a subset of languages from wordlist.
#     # Get temporary filename and output to that selecting on languages and donors.
#     # Input the temporary file as a wordlist and return the wordlist.
#     if not wordlist: return wordlist  # Leave as is.
#     languages_ = list()
#     if languages:
#         if languages == 'all': languages = wordlist.cols
#         languages_.extend(languages if isinstance(languages, list) else [languages])
#     if donors:
#         languages_.extend(donors if isinstance(donors, list) else [donors])
#     languages_ = list(set(languages_))
#     if not languages_: return wordlist  # No languages selected so leave as is.
#
#     with tempfile.TemporaryDirectory() as tmp:
#         file_path = Path(tmp).joinpath('tempwordlist').as_posix()
#         wordlist.output('tsv', filename=file_path, subset=True,
#                         rows=dict(doculect=" in "+str(languages_)),
#                         ignore='all', prettify=False)
#         # Now read in again.
#         wordlist_ = Wordlist(file_path+'.tsv')
#         check_coverage(wordlist_)
#         return wordlist_
#
#
# def check_coverage(wl=None):
#     print(f"Wordlist has {wl.width} languages, and {wl.height} concepts in {len(wl)} words.")
#     for i in range(200, 0, -1):
#         if mutual_coverage_check(wl, i):
#             print(f"Minimum mutual coverage is at {i} concept pairs.")
#             break
#     print(f"Average coverage is at {average_coverage(wl):.2f}")
#
#
# def compose_wl_from_cldf():
#     wl = Wordlist.from_cldf(
#         sabor().cldf_dir / "cldf-metadata.json",
#         columns=["language_id",
#                  "language_family",
#                  "concept_name",  # From concept relation, name field.
#                  "concept_id",  # From concept relation, id field.
#                  "value",
#                  "form",
#                  "segments",
#                  "borrowed",
#                  "borrowed_score",
#                  "donor_language",
#                  "donor_value"],
#         namespace=(('language_id', 'language'),
#                    ('language_family', 'family'),
#                    ("concept_id", "concept"),
#                    ('segments', 'tokens'))
#     )
#     return wl
#
#
# def compose_wl():
#     wl = compose_wl_from_cldf()
#     wl.add_entries('concept_name', 'concept_name',
#                    lambda x: x.lower(), override=True)
#     return wl
#
#
# def get_wordlist(filename=None):
#     if filename:
#         filepath = Path("foreign").joinpath(filename+".tsv").as_posix()
#         print(f"Foreign wordlist - file path: {filepath}", file=sys.stderr)
#         wl = Wordlist(filepath)
#     else:
#         wl = compose_wl()
#     return wl
#
#
# # Get dictionary of family:language from wordlist.
# def get_language_family(wl):
#     families = {}
#     for (ID, language, family) in wl.iter_rows('doculect', 'family'):
#         families[language] = family
#
#     return families
#

def run(args):
    ...
