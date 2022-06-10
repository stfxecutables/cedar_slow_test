import os
import stat
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, TypeVar, Union

# Comment out more of these to speed up execution
import numpy
import pytest
import scipy
import torch
import tqdm
from sklearn.model_selection import ParameterGrid
from torchvision.models import VGG, EfficientNet, ResNet
from torchvision.transforms import TrivialAugmentWide
from typing_extensions import Literal

ROOT = Path(__file__).resolve().parent

VisionClassifier = Union[ResNet, VGG, EfficientNet]
Augmentation = Union[None, TrivialAugmentWide]

T = TypeVar("T", bound="ArgEnum")


class ArgEnum(Enum):
    @classmethod
    def choices(cls) -> str:
        info = " | ".join([e.value for e in cls])
        return f"< {info} >"

    @classmethod
    def choicesN(cls) -> str:
        info = " | ".join([e.value for e in cls])
        return f"< {info} | None >"

    @classmethod
    def parse(cls, s: str) -> "ArgEnum":
        return cls(s.lower())

    @classmethod
    def parseN(cls, s: str) -> Optional["ArgEnum"]:
        if s.lower() in ["none", ""]:
            return None
        return cls(s.lower())

    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def names(cls) -> List[str]:
        return [e.name for e in cls]


class VisionDataset(ArgEnum):
    """
    FashionMnist = "fmnist"
    Cifar10 = "cifar10"
    Cifar100 = "cifar100"
    """

    FashionMnist = "fmnist"
    Cifar10 = "cifar10"
    Cifar100 = "cifar100"


class VisionArchitecture(ArgEnum):
    """
    EffNetB0 = "effnetb0"
    EffNetB7 = "effnetb7"
    Resnet18 = "resnet18"
    Resnet50 = "resnet50"
    VGG11 = "vgg11"
    """

    EffNetB0 = "effnetb0"
    EffNetB7 = "effnetb7"
    Resnet18 = "resnet18"
    Resnet50 = "resnet50"
    VGG11 = "vgg11"


class VisionAugment(ArgEnum):
    """
    CutMix = "cutmix"
    RandAugment = "randaugment"
    """

    CutMix = "cutmix"
    TrivialAugment = "trivialaug"


def ensure_dir(path: Path) -> Path:
    if path.exists():
        if path.is_dir():
            return path
        else:
            raise FileExistsError("Specified path exists and is a file, not directory")
    for parent in path.parents:
        if not parent.exists():
            raise FileNotFoundError(
                f"The parent directory {parent} does not exist. You have mis-specified a path constant or argument."
            )
    # exist_ok=True in rare case of TOCTOU / parallelism
    path.mkdir(exist_ok=True, parents=False)
    return path


Alloc = Literal["def", "rrg"]

CC_CLUSTER = os.environ.get("CC_CLUSTER")
USER = os.environ.get("USER")
MAIL = f"{USER}@stfx.ca"
PROJECT = Path(__file__).resolve().parent.parent
HTUNE_FINETUNE_SCRIPT = PROJECT / "experiments/finetune/htune_fine.py"

if CC_CLUSTER is None:
    CC_CLUSTER = "local"

SCRIPT_OUTDIR = (
    ensure_dir(ROOT / "test_job_scripts")
    if CC_CLUSTER == "local"
    else ensure_dir(ROOT / "job_scripts")
)
SCRIPT_BATCH_OUTDIR = ensure_dir(SCRIPT_OUTDIR / "batched")
SLURM_LOGS_OUTDIR = ensure_dir(ROOT / "slurm_logs")
SCRIPTS_DIR = ensure_dir(ROOT / "scripts")

RUNTIME = "00-03:00:00"
MIN_MEM = "32G"

LR_FIND_FINETUNE_BATCH_SCRIPT = f"""#!/bin/bash
#SBATCH --account=rrg-jlevman
#SBATCH --time=00-11:58:00
#SBATCH --signal=INT@300
#SBATCH --job-name=lrfind_fine
#SBATCH --output="/scratch/dberger/reproducible_dl/slurm_logs/lrfind_fine__%j_%u.out"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

PROJECT={PROJECT}

echo "Setting up python venv"
cd $PROJECT
source $PROJECT/.venv/bin/activate
PYTHON=$(which python)

echo "Job starting at $(date)"
{{cmds}}
echo "Job done at $(date)"

"""

LR_FIND_FINETUNE_BATCH_CMD = f"$PYTHON {HTUNE_FINETUNE_SCRIPT} {{args}}"


def chmod_plusx(file: os.PathLike) -> None:
    if not Path(file).exists():
        raise FileNotFoundError(f"No file: {file}")
    st = os.stat(file)
    os.chmod(file, st.st_mode | stat.S_IEXEC)


def generate_lrfind_finetune_jobscript() -> Path:
    """
    Notes
    -----
    Estimated runtime: slowest is EffNetb7 (obviously). Takes about 12min on the QuadroM5000, so
    probably more like 5-6 min on the v100. ResNet50 is about 6min on the Quadro, so likely 2-3
    on V100. Others are much much less, we can just estimate 2 minutes each. There are 5 models
    total, per iteration, so 6 + 3 + 3*2 = 15. There are 60 combinations total, grouping by 5 is
    12, so 12 * 15min = 3hrs. We can estimate 11:59 hours to be extremely safe, and since there is
    no cost to over-estimation besides wait-times, and this gets us under the 12hr threshold.
    """
    N_REPS = 1
    arguments = list(
        ParameterGrid(
            {
                "--model": VisionArchitecture.values(),
                "--dataset": VisionDataset.values(),
                "--augmentation": [VisionAugment.TrivialAugment.value, None],
                "--exp_number": [3, 6],
            }
        )
    )
    const_args = " --seed=None --lr_init=3e-4 --weight_decay=1e-5 --lr_find"
    commands = []
    for args in arguments:
        argstrs = []
        for arg, argval in args.items():
            argstrs.append(f"{arg}={argval}")
        argstr = " ".join(argstrs)
        argstr = argstr + const_args
        command = LR_FIND_FINETUNE_BATCH_CMD.format(args=argstr)
        for _ in range(N_REPS):
            commands.append(command)
    cmds = "\n".join(commands)
    script = LR_FIND_FINETUNE_BATCH_SCRIPT.format(cmds=cmds)
    outfile: Path = SCRIPT_OUTDIR / "submit_lr_find_all.sh"

    # We would normally do below, but I won't for the test:
    # with open(outfile, "w") as handle:
    #     handle.write(script)
    # print(f"Wrote script to {outfile}")
    # chmod_plusx(outfile)

    return outfile


if __name__ == "__main__":
    generate_lrfind_finetune_jobscript()
