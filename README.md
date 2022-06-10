# A Set of Scripts to Reproduce Cedar Slowness Issues

To run, clone the repo to somewhere on Cedar `$SCRATCH`, and `cd`. Then

```bash
time bash install.sh
```

All this does is create a virtual environment in `.venv` and install `torch` and `torchvision`,
along with some other extremely standard scientific python libs. It can take anywhere from 5-20
minutes to install the .venv on Cedar (likely because many files must be written during this
process). This is also evidenced in that setup tends to get stuck for a long time at:

```
Installing collected packages: iniconfig, typing_extensions, tqdm, tomli, threadpoolctl, pyparsing, py, pluggy, pillow-simd, numpy, joblib, attrs, torch, scipy, packaging, torchvision, scikit-learn, pytest
```

If you try the same install script on Graham, `.venv` install will almost always be done in <5 minutes

The subsequent `tar -cf venv.tar .venv` archiving will run in about a minute or two on Graham, but
could take forever (1+ hour) on Cedar. Again, running `tar` only requires many file reads, so this
indicates the problem is there.

Before running the next code, feel free to read through `really_slow.py`. It does basically nothing,
and would run in less than a second on your local machine. On Cedar, runtimes can be anywhere from 5
to 20 minutes. If you comment out unused imports, it runs faster, on average, but the more imports
you have, the slower it goes. You can see time estimates by running:

```bash
bash slow_script.sh
```

You should see output like:

```
```

To see how random this is, just run a bash / zsh loop to repeat this:

```bash
for x in 1 2 3 4 5 6 7 8 9 10; do bash slow_script.sh; done;
```

The results are highly random and depend partly on the time of day, but here is a sample output of
the above, when run on a login-node:

```bash
real	1m57.432s
user	0m2.235s
sys	0m1.133s

real	0m3.642s
user	0m1.982s
sys	0m0.811s

real	0m3.640s
user	0m2.066s
sys	0m0.793s

real	0m2.841s
user	0m1.656s
sys	0m0.619s

real	0m3.423s
user	0m2.048s
sys	0m0.770s

real	2m25.478s
user	0m2.325s
sys	0m1.195s

real	2m43.339s
user	0m2.237s
sys	0m1.228s

real	0m3.731s
user	0m1.979s
sys	0m0.825s

real	0m3.302s
user	0m1.809s
sys	0m0.742s

real	0m3.277s
user	0m1.846s
sys	0m0.755s

```

You can try this in a Cedar interactive-job too, and the results will be just as bad, maybe worse:

```
$ salloc --time=1:0:0 --ntasks=4 --mem-per-cpu=16G --account=def-jlevman

  salloc: Pending job allocation 36126573
  salloc: job 36126573 queued and waiting for resources
  salloc: job 36126573 has been allocated resources
  salloc: Granted job allocation 36126573
  salloc: Waiting for resource configuration
  salloc: Nodes cdr861 are ready for job

[dberger@cdr861 cedar_slow_test]$ for x in 1 2 3 4 5 6 7 8 9 10; do bash slow_script.sh; done;

  real	4m47.400s
  user	0m2.383s
  sys	0m2.722s

  real	4m56.302s
  user	0m2.306s
  sys	0m2.699s

  real	5m44.728s
  user	0m2.284s
  sys	0m2.544s

  real	6m50.440s
  user	0m2.253s
  sys	0m2.730s

  real	7m38.160s
  user	0m2.182s
  sys	0m2.409s

  real	7m52.240s
  user	0m2.204s
  sys	0m2.439s

  real	7m40.668s
  user	0m2.257s
  sys	0m2.292s

  real	7m27.859s
  user	0m2.264s
  sys	0m2.531s

  real	5m50.572s
  user	0m2.239s
  sys	0m2.389s
  salloc: Job 36126573 has exceeded its time limit and its allocation has been revoked

  slurmstepd: error: *** STEP 36126573.interactive ON cdr861 CANCELLED AT 2022-06-10T09:54:33 DUE TO TIME LIMIT ***
  srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
  srun: error: cdr861: task 0: Killed
  srun: launch/slurm: _step_signal: Terminating StepId=36126573.interactive
  srun: error: Timed out waiting for job step to complete
```

Maybe this is because in the interactive job we have to reach across to the login node to get
our `.venv` files? But we can test this is not the case in an interactive job with `still_slow.sh`:

```
$ salloc --time=3:0:0 --ntasks=1 --mem-per-cpu=16G --account=def-jlevman

  salloc: Pending job allocation 36134385
  salloc: job 36134385 queued and waiting for resources
  salloc: job 36134385 has been allocated resources
  salloc: Granted job allocation 36134385

[dberger@cdr861 cedar_slow_test]$ bash still_slow.sh

  real	0m3.899s
  user	0m0.268s
  sys	0m3.014s
  Un-tarred .venv to temp dir. Time elapsed above.
  Copied single script file to temp dir. Time elapsed above.
  Now code is setup all in the temp directory:
  total 8.0K
  drwxr-x---  3 dberger dberger   41 Jun 10 10:14 .
  drwxr-xr-x 11 root    root     236 Jun 10 10:14 ..
  -rw-r-----  1 dberger dberger 5.9K Jun 10 10:14 really_slow.py
  drwxr-x---  4 dberger dberger   46 Jun 10 08:37 .venv

  real	8m29.602s
  user	0m1.898s
  sys	0m2.300s

  real	7m6.051s
  user	0m1.881s
  sys	0m2.490s

  real	6m8.209s
  user	0m1.842s
  sys	0m2.184s

  real	7m37.467s
  user	0m1.778s
  sys	0m2.081s

  real	6m24.056s
  user	0m1.877s
  sys	0m2.119s

  real	9m26.496s
  user	0m1.965s
  sys	0m2.275s

  real	6m47.930s
  user	0m1.957s
  sys	0m2.300s

^CTraceback (most recent call last):
  File "really_slow.py", line 9, in <module>
    import numpy
  File "/scratch/dberger/cedar_slow_test/.venv/lib/python3.8/site-packages/numpy/__init__.py", line 155, in <module>
    from . import random
  File "/scratch/dberger/cedar_slow_test/.venv/lib/python3.8/site-packages/numpy/random/__init__.py", line 180, in <module>
    from . import _pickle
  File "/scratch/dberger/cedar_slow_test/.venv/lib/python3.8/site-packages/numpy/random/_pickle.py", line 6, in <module>
    from ._generator import Generator
  File "<frozen importlib._bootstrap>", line 389, in parent
KeyboardInterrupt

# I terminated the job here, the point has been made. As usual, we are stuck in imports.
```

If anything, the runtimes even look worse.

By contrast, on Graham, results of the `still_slow.sh` test are fast, as expected:

```
$ salloc --time=1:0:0 --ntasks=1 --mem-per-cpu=16GB --account=def-jlevman

  salloc: Pending job allocation 61912016
  salloc: job 61912016 queued and waiting for resources
  salloc: job 61912016 has been allocated resources
  salloc: Granted job allocation 61912016
  salloc: Waiting for resource configuration
  salloc: Nodes gra1136 are ready for job

[dberger@gra1136 cedar_slow_test]$ bash still_slow.sh

  real	0m3.089s
  user	0m0.206s
  sys	0m2.861s
  Un-tarred .venv to temp dir. Time elapsed above.
  Copied single script file to temp dir. Time elapsed above.
  Now code is setup all in the temp directory:
  total 20K
  drwxr-x---  3 dberger dberger 4.0K Jun 10 12:38 .
  drwxr-xr-x 36 root    root    4.0K Jun 10 12:37 ..
  -rw-r-----  1 dberger dberger 5.9K Jun 10 12:38 really_slow.py
  drwxr-x---  4 dberger dberger 4.0K Jun 10 12:22 .venv

  real	0m53.129s
  user	0m2.344s
  sys	0m1.873s

  real	0m2.657s
  user	0m1.467s
  sys	0m0.584s

  real	0m2.546s
  user	0m1.382s
  sys	0m0.576s

  real	0m2.524s
  user	0m1.405s
  sys	0m0.551s

  real	0m2.545s
  user	0m1.449s
  sys	0m0.517s

  real	0m2.573s
  user	0m1.435s
  sys	0m0.563s

  real	0m2.550s
  user	0m1.406s
  sys	0m0.563s

  real	0m2.530s
  user	0m1.418s
  sys	0m0.533s

  real	0m2.573s
  user	0m1.424s
  sys	0m0.550s

  real	0m2.633s
  user	0m1.426s
  sys	0m0.543s
```