# Borrowing in South American Languages

## How to cite

If you use these data please cite
this dataset using the DOI of the [particular released version](../../releases/) you were using

## Description


This dataset is licensed under a CC-BY-4.0 license

## Statistics


![Glottolog: 100%](https://img.shields.io/badge/Glottolog-100%25-brightgreen.svg "Glottolog: 100%")
![Concepticon: 100%](https://img.shields.io/badge/Concepticon-100%25-brightgreen.svg "Concepticon: 100%")
![Source: 0%](https://img.shields.io/badge/Source-0%25-red.svg "Source: 0%")
![BIPA: 100%](https://img.shields.io/badge/BIPA-100%25-brightgreen.svg "BIPA: 100%")
![CLTS SoundClass: 100%](https://img.shields.io/badge/CLTS%20SoundClass-100%25-brightgreen.svg "CLTS SoundClass: 100%")

- **Varieties:** 9
- **Concepts:** 1,308
- **Lexemes:** 13,783
- **Sources:** 0
- **Synonymy:** 1.29
- **Invalid lexemes:** 0
- **Tokens:** 82,197
- **Segments:** 136 (0 BIPA errors, 0 CTLS sound class errors, 136 CLTS modified)
- **Inventory size (avg):** 41.00

## Possible Improvements:



- Entries missing sources: 13783/13783 (100.00%)

## CLDF Datasets

The following CLDF datasets are available in [cldf](cldf):

- CLDF [Wordlist](https://github.com/cldf/cldf/tree/master/modules/Wordlist) at [cldf/cldf-metadata.json](cldf/cldf-metadata.json)


## Example Commands


Modules include multiple, pairwise, multiple with lingrex, and reportmultiple.

Each module provides help as shown here.

<code>% cldfbench sabor.pairwise --help</code>


Pairwise tests pairwise relationships between language forms and writes multiple words distance and words status reports to output as well as summary perfomance to stdout.

<code>% cldfbench sabor.pairwise --threshold .3 .4 .5 > output/pairwise-log.txt</code>

Multiple test multiple relationships between language forms at the same time and writes multiple words status reports to output, saves analysis file in store, as well writes summary performance to stdout.

<code>% cldfbench sabor.multiple --threshold .5 .6 .7 > output/multiple-log.txt</code>

Lingrex performs a draft common morpheme analysis and saves the lingrex analysis file in store.

<code>% python saborcommands/multiple.py --module lingrex</code>

Report multiple is available as a follow-on reporting to multiple in the event the user wants to change reporting options without having to run the initial analysis over again.