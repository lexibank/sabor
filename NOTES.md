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

