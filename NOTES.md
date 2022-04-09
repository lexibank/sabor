# Command examples

```
$ cldfbench sabor.pairwise --threshold .3 .4 .5 > output/pairwise-log.txt
```

Writes multiple words distance and words status (largely redundant with distance) reports to output.

```
$ cldfbench sabor.multiple --threshold .5 .6 .7 > output/multiple-log.txt
```

Writes multiple words status reports to output and saves analysis file in store.

```
$ cldfbench sabor.analyzelingrex > output/lingrex-log.txt
```
Writes lingrex analysis file in store.  
Arguments for internal and external thresholds.


```
$ cldfbench sabor.reportlingrex > output/lingrex-report-log.txt
```
Writes lingrex words status report file and log to output.
