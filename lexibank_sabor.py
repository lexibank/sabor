import pathlib
import zipfile
import itertools
import collections

import pycldf
from cldfbench import CLDFSpec
from pylexibank import Dataset as BaseDataset
from cltoolkit import Wordlist as CLWordlist
from cldfzenodo import oai_lexibank
from pyclts import CLTS
from git import Repo, GitCommandError
from csvw.dsv import reader
import lingpy
from clldutils.misc import slug
from tabulate import tabulate
from pathlib import Path


from pylexibank import Concept, Lexeme, progressbar, FormSpec
import attr
from csvw.dsv import UnicodeWriter
import json



@attr.s
class CustomLexeme(Lexeme):
    Value_in_Source = attr.ib(default=None)
    ConceptInSource = attr.ib(default=None)
    Borrowed = attr.ib(default=None)
    Borrowed_Score = attr.ib(default=None)
    Borrowed_Base = attr.ib(default=None)
    Age = attr.ib(default=None)
    Age_Score = attr.ib(default=None)
    Source_Language = attr.ib(default=None)
    Source_Meaning = attr.ib(default=None)
    Source_Word = attr.ib(default=None)



class Dataset(BaseDataset):
    dir = pathlib.Path(__file__).parent
    id = "sabor"
    lexeme_class = CustomLexeme
    form_spec = FormSpec(
            replacements=[(" ", "_")], 
            separators="~;,/", missing_data=["∅"], first_form_only=True)

    def cmd_download(self, args):
        
        try:
            repo = Repo.clone_from(
                    "https://github.com/lexibank/wold.git",
                    self.raw_dir / "wold")
            repo.git.checkout("v4.0")
        except:
            args.log.info("clone failed, repository already downloaded")
        try:
            repo = Repo.clone_from(
                    "https://github.com/intercontinental-dictionary-series/ids.git",
                    self.raw_dir / "ids")
            repo.git.checkout("v4.2")
        except:
            args.log.info("clone failed, repository already downloaded")

    def cmd_makecldf(self, args):
        
        borrowings = {
                form.data["Target_Form_ID"]: (
                    form.data["Source_languoid"],
                    form.data["Source_meaning"],
                    form.data["Source_word"]
                    ) for form in pycldf.Dataset.from_metadata(
                        self.raw_dir / "wold" / "cldf" / "cldf-metadata.json").objects(
                            "BorrowingTable")}
        args.log.info("loaded borrowings")

        languages = {}
        for language in self.languages:
            languages[language["Name"]] = language

        wl = CLWordlist(
                [
                    pycldf.Dataset.from_metadata(
                        self.raw_dir / ds / "cldf" / "cldf-metadata.json") for ds in ["wold", "ids"]
                    ], 
                ts=args.clts.api.bipa)

        concepts = {}

        ids_languages = {
                "ids-Portuguese": wl.languages["ids-178"], 
                "ids-Spanish": wl.languages["ids-176"]
                }
        concepts = {}
        for name, language in ids_languages.items():
            args.writer.add_language(
                    ID=name[4:],
                    Name=language.name,
                    Glottocode=language.glottocode,
                    Latitude=language.latitude,
                    Longitude=language.longitude)
            for concept in language.concepts:
                if concept.concepticon_gloss not in concepts:
                    concepts[concept.concepticon_gloss] = concept
                    args.writer.add_concept(
                            ID=slug(concept.id, lowercase=False),
                            Name=concept.concepticon_gloss,
                            Concepticon_ID=concept.concepticon_id,
                            Concepticon_Gloss=concept.concepticon_gloss
                            )
            for form in language.forms:
                args.writer.add_forms_from_value(
                        Language_ID=name[4:],
                        Parameter_ID=slug(form.concept.id, lowercase=False),
                        Value=form.form,
                        Value_in_Source=form.value,
                        Local_ID=form.id,
                        )
        args.log.info("added spa and pt")

        wold_languages = {}
        for language in wl.languages:
            if language.name in languages and language.id.startswith('wold-'):
            # Some languages are in WOLD both as wold receiver and ids donor languages.
                wold_languages[language.name] = language
        for name, language in wold_languages.items():
            print(f"Added: name {name}, language {language}")
            args.writer.add_language(
                    ID=language.id[5:],  # Drop the wold- prefix.
                    Name=language.name,
                    Glottocode=language.glottocode,
                    Latitude=language.latitude,
                    Longitude=language.longitude)
            for form in language.forms_with_sounds:
                if form.concept and form.concept.concepticon_gloss in concepts:
                    args.writer.add_form_with_segments(
                            Local_ID=form.id,
                            Language_ID=form.language.id[5:],
                            Parameter_ID=slug(form.concept.id, lowercase=False),
                            # Borrowed=form.data["Borrowed"],
                            # Original form had 5 point Likert type scale of borrowing likelihood.
                            Borrowed_Score=form.data["Borrowed_score"], 
                            # 0.0 not borrowed ~ 1.0 = borrowed
                            Borrowed=float(form.data["Borrowed_score"]) > 0.90,
                            Borrowed_Base=form.data["borrowed_base"],  
                            # This sometimes includes source word.
                            Value=form.value,
                            Form=form.form,
                            Segments=form.sounds,
                            Age=form.data["Age"],
                            Age_Score=form.data["Age_score"],
                            Source_Language=borrowings.get(form.id[5:], [""])[0],
                            Source_Meaning=borrowings.get(form.id[5:], ["", ""])[1],
                            Source_Word=borrowings.get(form.id[5:], ["","",""])[2]
                            )

