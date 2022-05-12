import pathlib

import pycldf
from pylexibank import Dataset as BaseDataset
from cltoolkit import Wordlist as CLWordlist
from git import Repo, GitCommandError
import lingpy
from clldutils.misc import slug


from pylexibank import Concept, Lexeme, progressbar, FormSpec
import attr

import collections


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
                # "ids-Portuguese": wl.languages["ids-178"],
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
            "donor_language", "donor_value"],
    )
    # donor_language and donor_value fields read as None when empty.
    for idx in wl:
        if wl[idx, "donor_language"] is None: wl[idx, "donor_language"] = ""
        if wl[idx, "donor_value"] is None: wl[idx, "donor_value"] = ""
    return wl


def sds_by_concept(donors, targets, func, threshold):
    """
    Function applies the pairwise comparison and returns a dictionary with \
            the results per item."""

    # hits is a dictionary with target ID as key and list of possible donor
    # candidate ids as value 
    hits = collections.defaultdict(list)
    for idxA, tksA in donors.items():
        for idxB, tksB in targets.items():
            score = func(tksA, tksB)
            if score < threshold:
                hits[idxB] += [(idxA, score)]
    # we sort the hits, as we can have only one donor
    out = {}
    for hit, pairs in hits.items():
        out[hit] = sorted(pairs, key=lambda x: x[1])[0][0]
    return out


def simple_donor_search(
        wordlist,
        donors,
        family="language_family",
        concept="concept",
        segments="tokens",
        donor_lng="source_language",
        donor_id="source_id",
        func=None,
        threshold=0.45,
        **kw
        ):
    """
    Find borrowings by carrying out a pairwise comparison of donor and target words.

    :param wordlist: LingPy wordlist.
    :param donors: Donor languages, passed as a list.
    :param family: Column in which language family information is given in the wordlist.
    :param concept: Column in which concept information is given in the wordlist.
    :param segments: Column in which segmented IPA tokens are given in the wordlist.
    :param donor_lng: Column to which information on predicted donor languages will
      be written (defaults to "source_language").
    :param donor_id: Column to which we write information on the ID of the predicted donor.
    :param func: Function comparing two sequences and returning a distance
      score (defaults to sca_distance).
    :param threshold: Threshold, at which we recognize a word as being borrowed.
    """
    func = func or sca_distance

    # get concept slots from the data (in case we use broader concepts by clics
    # communities), we essentially already split data in donor indices and
    # target indices by putting them in a list.
    donor_families = {fam for (ID, lang, fam)
                      in wordlist.iter_rows('doculect', family)
                      if lang in donors}

    concepts = {concept: [[], []] for concept in set(
        [wordlist[idx, concept] for idx in wordlist])}
    for idx in wordlist:
        if wordlist[idx, "doculect"] in donors:
            concepts[wordlist[idx, concept]][0] += [idx]
        # languages from donor families are not target languages.
        elif wordlist[idx, family] not in donor_families:
            concepts[wordlist[idx, concept]][1] += [idx]

    # iterate over concepts and identify potential borrowings
    B = {idx: 0 for idx in wordlist}
    for concept, (donor_indices, target_indices) in progressbar(concepts.items(), 
            desc="searching for borrowings"):
        # hits is a dictionary with target ID as key and list of possible donor
        # candidate ids as value 
        hits = sds_by_concept(
                {idx: wordlist[idx, segments] for idx in donor_indices},
                {idx: wordlist[idx, segments] for idx in target_indices},
                func,
                threshold
                )
        # we sort the hits, as we can have only one donor
        for hit, prediction in hits.items():
            B[hit] = prediction

    wordlist.add_entries(
            donor_lng, B, lambda x: wordlist[x, "doculect"] if x != 0 else "")
    wordlist.add_entries(
            donor_id, B, lambda x: x if x != 0 else "")


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


def evaluate_borrowings_fs(wordlist, pred, gold, donors, donor_families, family="family"):
    """
    Return F-Scores for the donor detection.
    """
    # Defensive programming:
    # Check for None and be sure of pred versus gold.
    # Return F1 score overall.
    # Evaluation wordlist is from parent.
    fn = fp = tn = tp = 0
    for idx, pred_lng, gold_lng in wordlist.iter_rows(pred, gold):
        if wordlist[idx, family] not in donor_families:
            if not pred_lng:
                if not gold_lng: tn += 1
                elif gold_lng in donors: fn += 1
                else: tn += 1
            elif pred_lng:
                if not gold_lng: fp += 1
                elif gold_lng in donors: tp += 1
                else: fp += 1
    return tp/(tp + (fp + fn)/2)

