sbatch evaluation_sbatch.sh encoderDecoder_evaluate_script.py hyperparameters/52273944_encoderDecoder_hyperparameters.py ../data/eval_topX_recall_at_10_415.csv ../models/enc-dec_suomi24_2001/52273944/800000_checkpoint.tar ../models/model.bin 
sleep 240
sbatch evaluation_sbatch.sh encoderDecoder_evaluate_script.py hyperparameters/52274668_encoderDecoder_hyperparameters.py ../data/eval_topX_recall_at_10_415.csv ../models/enc-dec_opensubs/52274668/800000_checkpoint.tar ../models/model.bin       
sleep 240
sbatch evaluation_sbatch.sh transformer_evaluate_script.py hyperparameters/52359009_transformer_hyperparameters.py ../data/eval_topX_recall_at_10_415.csv ../models/transformer_opensubs/52359009/1600000_checkpoint.tar ../models/model.bin
sleep 240
sbatch evaluation_sbatch.sh transformer_evaluate_script.py hyperparameters/52359148_transformer_hyperparameters.py ../data/eval_topX_recall_at_10_415.csv ../models/transformer_suomi24/52359148/1600000_checkpoint.tar ../models/model.bin
sleep 240
sbatch evaluation_sbatch.sh encoderDecoder_evaluate_script.py hyperparameters/52273944_encoderDecoder_hyperparameters.py ../data/eval_1k_suomi24_2002.csv ../models/enc-dec_suomi24_2001/52273944/800000_checkpoint.tar ../models/model.bin 
sleep 240
sbatch evaluation_sbatch.sh encoderDecoder_evaluate_script.py hyperparameters/52274668_encoderDecoder_hyperparameters.py ../data/eval_1k_opensubs.csv ../models/enc-dec_opensubs/52274668/800000_checkpoint.tar ../models/model.bin       
sleep 240
sbatch evaluation_sbatch.sh transformer_evaluate_script.py hyperparameters/52359009_transformer_hyperparameters.py ../data/eval_1k_opensubs.csv ../models/transformer_opensubs/52359009/1600000_checkpoint.tar ../models/model.bin
sleep 240
sbatch evaluation_sbatch.sh transformer_evaluate_script.py hyperparameters/52359148_transformer_hyperparameters.py ../data/eval_1k_suomi24_2002.csv ../models/transformer_suomi24/52359148/1600000_checkpoint.tar ../models/model.bin
