#!/bin/bash -l

#SBATCH --partition dgx-spa,dgx-common
#SBATCH --time=110:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/work/jpleino1/log/s2s_chatbot-%j.log
#SBATCH -e /scratch/work/jpleino1/log/s2s_chatbot-%j.log

export=PATH,HOME,USER,TERM,WRKDIR
export PYTHONUNBUFFERED=1

module purge
module load anaconda3
source activate /scratch/work/jpleino1/conda/envs/nmt_chatbot
#conda activate nmt_chatbot
script=$1

git log | head -n 6

echo "encoderDecoder_main.py"
cat encoderDecoder_main.py

echo "encoderDecoder_hyperparameters.py"
cat encoderDecoder_hyperparameters.py
cat encoderDecoder_hyperparameters.py > "$SLURM_JOBID"_encoderDecoder_hyperparameters.py

echo "encoderDecoder_global_variables.py"
cat encoderDecoder_global_variables.py

echo "encoderDecoder_prep_data.py"
cat encoderDecoder_prep_data.py

echo "encoderDecoder_voc.py"
cat encoderDecoder_voc.py

echo "encoderDecoder_training.py"
cat encoderDecoder_training.py

echo "encoderDecoder_models.py"
cat encoderDecoder_models.py

echo "encoderDecoder_sbatch.sh"
cat encoderDecoder_sbatch.sh

echo "encoderDecoder_evaluation.py"
cat encoderDecoder_evaluation.py

echo "encoderDecoder_evaluate_script.py"
cat encoderDecoder_evaluate_script.py

python $script $SLURM_JOBID
