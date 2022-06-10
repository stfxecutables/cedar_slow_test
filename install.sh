rm -rf .venv
module load python/3.8.10
virtualenv --no-download .venv
source .venv/bin/activate
pip install --upgrade pip
time pip install torch torchvision numpy scikit-learn scipy typing_extensions tqdm pytest
echo "Archiving .venv to venv.tar for later testing..."
time tar -cf venv.tar .venv
echo "Done setup. Time to archive .venv above"