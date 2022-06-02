import pathlib
import collections

import pycldf
from pylexibank import Dataset as BaseDataset
from cltoolkit import Wordlist as CLWordlist
from git import Repo, GitCommandError
import lingpy
from lingpy import (Wordlist, LexStat)
from clldutils.misc import slug

from pylexibank import Lexeme, Language, FormSpec, Concept
import attr

BOR_CRITICAL_VALUE = 0.67


@attr.s
class CustomLanguage(Language):
    Spanish_Borrowings = attr.ib(default=None)



@attr.s
class CustomLexeme(Lexeme):
    Value_in_Source = attr.ib(default=None)
    ConceptInSource = attr.ib(default=None)
    Borrowed = attr.ib(default=None)
    Borrowed_Score = attr.ib(default=None)
    Borrowed_Base = attr.ib(default=None)
    Age = attr.ib(default=None)
    Age_Score = attr.ib(default=None)
    Donor_Language = attr.ib(default=None)
    Donor_Meaning = attr.ib(default=None)
    Donor_Value = attr.ib(default=None)


class Dataset(BaseDataset):
    dir = pathlib.Path(__file__).parent
    id = "sabor"
    lexeme_class = CustomLexeme
    language_class = CustomLanguage
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

        ids_languages = {
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
        args.log.info("added {}".format(list(ids_languages.keys())))

        wold_languages = {}
        for language in wl.languages:
            if language.name in languages and language.id.startswith('wold-'):
                # Some languages are in WOLD both as wold receiver and ids donor languages.
                wold_languages[language.name] = language
        for name, language in wold_languages.items():
            print("Added: name {name}, language {language}".format(
                name=name, language=language.id))
            borrowed = sum(
                    [1 for form in language.forms_with_sounds if borrowings.get(
                        form.id[5:], [""])[0] == "Spanish"])
            args.writer.add_language(
                    ID=language.id[5:],  # Drop the wold- prefix.
                    Name=language.name,
                    Glottocode=language.glottocode,
                    Latitude=language.latitude,
                    Longitude=language.longitude,
                    Spanish_Borrowings=borrowed/len(language.forms_with_sounds))
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
                            Donor_Language=borrowings.get(form.id[5:], [""])[0],
                            Donor_Meaning=borrowings.get(form.id[5:], ["", ""])[1],
                            Donor_Value=borrowings.get(form.id[5:], ["", "", ""])[2]
                            )


def get_our_wordlist():
    wl = lingpy.Wordlist.from_cldf(
        str(Dataset().cldf_dir / "cldf-metadata.json"),
        columns=[
            "language_id", "language_family",
            "concept_name", "value", "form", "segments",
            "borrowed_score", "donor_language", "donor_value"],
    )
    # donor_language and donor_value fields read as None when empty.
    for idx in wl:
        if wl[idx, "donor_language"] is None: wl[idx, "donor_language"] = ""
        if wl[idx, "donor_value"] is None: wl[idx, "donor_value"] = ""
        # Only trust donor language and value if near certain.
        borrowed_score = wl[idx, "borrowed_score"]
        if borrowed_score is None or \
                float(borrowed_score) < BOR_CRITICAL_VALUE:
            wl[idx, "donor_language"] = ""
            wl[idx, "donor_value"] = ""

    return wl


def sca_distance(seqA, seqB, **kw):
    """
    Shortcut for computing SCA distances from two strings.
    """

    pair = lingpy.Pairwise(seqA, seqB)
    pair.align(distance=True, **kw)

    return pair.alignments[0][-1]


def edit_distance(seqA, seqB, **kw):
    """
    Shortcut normalized edit distance.
    """
    return lingpy.edit_dist(seqA, seqB, normalized=True)


def evaluate_borrowings_fs(wordlist, pred, gold, donors):
    """
    Return F1-Score for the donor detection.
    """
    fs = evaluate_borrowings(wordlist, pred, gold, donors)  # , donor_families, family)
    return fs['f1']


def evaluate_borrowings(wordlist, pred, gold, donors, beta=1.0):
    """
    Return F1 and related scores for the donor detection.
    """
    # Check for None and be sure of pred versus gold.
    # Return tp, tn, fp, fn, precision, recall, f1score, accuracy.
    # Evaluation wordlist is from predict function of analysis method.
    fn = fp = tn = tp = 0

    for idx, pred_lng, gold_lng in wordlist.iter_rows(pred, gold):
        if wordlist[idx, 'doculect'] not in donors:
            if not pred_lng:
                if not gold_lng: tn += 1
                elif gold_lng in donors: fn += 1
                else: tn += 1
            elif pred_lng:
                if not gold_lng: fp += 1
                elif gold_lng in donors: tp += 1
                else: fp += 1

    precision = tp/(tp + fp) if tp + fp else 0
    recall = tp/(tp + fn) if tp + fn else 0
    f1 = tp/(tp + (fp + fn)/2) if tp + fp + fn else 0
    fb = (1.0 + beta**2.0) * (precision * recall) / \
        (beta**2.0 * precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn)/(tp + tn + fp + fn)

    return {'fn': fn, 'fp': fp, 'tn': tn, 'tp': tp,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'fb': round(fb, 3),
            'accuracy': round(accuracy, 3)}


def our_path(*comps):
    return Dataset().dir.joinpath(*comps).as_posix()


# Subset wordlist on selection of languages.
def subset_wl(wl, languages):
    hdr = ['ID'] + [key.upper() for key in wl.columns]
    new_wl = {0: hdr}
    for idx in wl.iter_rows():
        idx_ = idx[0]
        if wl[idx_, 'doculect'] in languages:
            new_wl[idx_] = [idx_] + wl[idx_]
    return LexStat(new_wl) if isinstance(wl, LexStat) else Wordlist(new_wl)


def get_language_list(languages, donors):
    languages = [languages] if isinstance(languages, str) else languages
    donors = [donors] if isinstance(donors, str) else donors
    return list(set(languages + donors))
