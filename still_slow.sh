#!/bin/bash
time tar -xf venv.tar -C $SLURM_TMPDIR
echo "Un-tarred .venv to temp dir. Time elapsed above."
cp really_slow.py $SLURM_TMPDIR
echo "Copied single script file to temp dir. Time elapsed above."
cd $SLURM_TMPDIR
echo "Now code is setup all in the temp directory:"
ls -lha
source .venv/bin/activate
for x in 1 2 3 4 5 6 7 8 9 10; do time python really_slow.py; done;