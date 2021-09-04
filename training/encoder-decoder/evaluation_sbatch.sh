#!/bin/bash -l

#SBATCH --partition batch
#SBATCH --time=4:00:00    # 6 hours
#SBATCH --mem-per-cpu=3000    # 1024MB of memory
#SBATCH -o /scratch/work/jpleino1/log/s2s_chatbot-eval_metrics-%j.log
#SBATCH -e /scratch/work/jpleino1/log/s2s_chatbot-eval_metrics-%j.log

export=PATH,HOME,USER,TERM,WRKDIR
export PYTHONUNBUFFERED=1

module purge
module load anaconda3
source activate /scratch/work/jpleino1/conda/envs/nmt_chatbot
#conda activate nmt_chatbot
script=$1
hyperparameters=$2
data=$3
model_tar=$4
morfessor_model=$5
git log | head -n 6

echo "python "$script" "$hyperparameters" "$data" "$model_tar" "$morfessor_model""
python $script $hyperparameters $data $model_tar $morfessor_model
