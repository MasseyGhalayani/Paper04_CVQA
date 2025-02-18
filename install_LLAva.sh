
cd || exit # To Go to Home dir

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA || exit


conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .


module load cuda/12.6.1


# moving files that are needed in LLaVA repo
cd || exit # To Go to Home dir
cp Paper04_CVQA/job.sbacth  LLaVA/job.sbatch
cp Paper04_CVQA/scripts/run_eval_llava.py LLaVA/run_eval_llava.py


