# Command examples

```commandline
$ cldfbench sabor.pairwise --threshold .3 .4 .5 > output/pairwise-log.txt
```

Writes multiple words distance and words status (largely redundant with distance) reports to output.

```commandline
$ cldfbench sabor.cluster --threshold .5 .6 .7 > output/cluster-log.txt
```

Writes multiple words status reports to output and saves analysis file in store.

```commandline
$ cldfbench sabor.analyzelingrex > output/lingrex-log.txt
```
Writes lingrex analysis file in store.  
Arguments for internal and external thresholds.


```commandline
$ cldfbench sabor.reportlingrex > output/lingrex-report-log.txt
```
Writes lingrex words status report file and log to output.


```commandline
% cldfbench sabor.lingrexfscore 
```

Calculates and reports F1 score for all languages with focus on Spanish and Portuguese donors 
using the existing lingrex-analysis.tsv data file.

```commandline
% cldfbench sabor.lingrexfscore --infile ''
```
Calculates and reports F1 score for all languages with focus on Spanish and Portuguese donors 
based on a new analysis of all languages using default LingRex analysis configuration
and using the default lingrex-analysis.tsv data file store for the analysis.

```commandline
% cldfbench sabor.lingrexfscore --anyloan
```
Calculates and reports F1 score for all languages without consideration for donor 
using the existing lingrex-analysis.tsv data file.

```commandline
% cldfbench sabor.lingrexfscore --infile '' --threshold 0.7 --ext_threshold 0.35
```
Calculates and reports F1 score for all languages with focus on Spanish and Portuguese donors 
based on a new analysis of all languages using designated internal and external thresholds,
and using the default lingrex-analysis.tsv data file store for the analysis.
Note: This analysis gives F1 Score of 0.787 - comparable to other results.

