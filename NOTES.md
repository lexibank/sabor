# Command examples

```
$ cldfbench sabor.pairwise --threshold .3 .4 .5 > output/pairwise-log.txt
```

Writes multiple words distance and words status (largely redundant with distance) reports to output.

```
$ cldfbench sabor.multiple --threshold .5 .6 .7 > output/multiple-log.txt
```

writes multiple words status reports to output and saves analysis file in store.

```
$ python saborcommands/multiple.py --module lingrex
```
saves lingrex analysis file in store.



