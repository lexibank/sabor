# Borrowing in South American Languages

## How to cite

If you use these data please cite
this dataset using the DOI of the [particular released version](../../releases/) you were using

## Description


This dataset is licensed under a CC-BY-4.0 license

## Notes

# Command examples

# Command examples

### Cluster module
```commandline
% cldfbench sabor.analyzecluster --threshold .5 .6 .7 > output/cluster-log.txt
```

Writes log file and saves analysis file in store.  
By default, borrowing detection is evaluated and results written to log output.

```commandline
 % cldfbench sabor.evaluatecluster --infile cluster > output/cluster-evaluation-log.txt
```
Writes evaluation to log file output.

```commandline
% cldfbench sabor.reportcluster --infile cluster > output/cluster-report-log.txt
```
Writes log file and detail diagnostic reports to output.

### LingRex module
```commandline
% cldfbench sabor.analyzelingrex --threshold 0.7 --ext_threshold 0.35 > output/lingrex-log.txt
```
Performs analysis, saves lingrex analysis in store, writes log to output.
By default, borrowing detection is evaluated and results written to log output.

```commandline
% cldfbench sabor.evaluatelingrex --infile lingrex-analysis > output/lingrex-evaluation-log.txt
```
Writes evaluation to log file output.

```commandline
% cldfbench sabor.reportlingrex --infile lingrex-analysis > output/lingrex-report-log.txt
```
Writes log file and detail diagnostic report to output.

### Pairwise module
```commandline
% cldfbench sabor.analyzepairwise > output/pairwise-log.txt 
```
Performs analysis, saves pairwise analysis to store, writes log to output.
By default, borrowing detection is evaluated and results written to log output.

```commandline
% cldfbench sabor.evaluatepairwise --infile pairwise.tsv > output/pairwise-evaluation-log.txt
```
Writes evaluatation to log file output.

```commandline
% cldfbench sabor.reportpairwise --infile pairwise.tsv > output/pairwise-report-log.txt
```
Writes log file and detail diagnostic report to output.

## Run Experiments

```commandline
 % python examples/run_cluster_exps.py --series scr-exp-1 > exp-results/cluster-scr-exp-1.log.txt
```
Run experimental design for cluster module with file  exp-scripts/cluster-scr-exp-1.tsv and
write results to exp-results/cluster-scr-exp-1-results.tsv




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
- **Segments:** 136 (0 BIPA errors, 0 CLTS sound class errors, 136 CLTS modified)
- **Inventory size (avg):** 41.00

## Possible Improvements:



- Entries missing sources: 13783/13783 (100.00%)

## CLDF Datasets

The following CLDF datasets are available in [cldf](cldf):

- CLDF [Wordlist](https://github.com/cldf/cldf/tree/master/modules/Wordlist) at [cldf/cldf-metadata.json](cldf/cldf-metadata.json)