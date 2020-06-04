# Automatic evaluation

This folder contains scripts for automatically evaluating the models.

## Evaluation of chatbot models:

Chatbot models were evaluated by eight different metrics: 
- hits@1/N
- N-choose-k
- cross-entropy averaged over sentences
- character-based cross-entropy averaged over sentences
- BLUE comparing words
- BLEU comparing morphs
- chrF comparing words
- chrF comparing morphs. 

Files:
- create_eval_set.py : Script to create an N-choose-k and hits@1/N evaluation set from a text file.
- encoderDecoder_evaluate_script.py : Script that downloads the proper model and calculates the automatic metrics for it for given evaluation file.
- encoderDecoder_evaluation.py : Contains the functions encoderDecoder_evaluate_script.py uses.
- eval_1k_opensubs.csv : N-choose-k csv file similar to eval_topX_recall_at_10_415.csv from opensubtitles data.
- eval_1k_suomi24_2002.csv : N-choose-k csv file similar to eval_topX_recall_at_10_415.csv from suomi24 data.
- eval_topX_recall_at_10_415.csv : N-choose-k csv file. More detailed description in human evalution folder.
- transformer_evaluate_script.py : Same as encoderDecoder_evaluate_script.py except for the transformer.
- transformer_evaluation.py : Same as encoderDecoder_evaluation.py except for the the transformer.
