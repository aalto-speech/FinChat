# FinChat

This README is for the development branch of the FinChat corpus which has been designed to be as replicable as possible. This is in comparison to the master branch which is currently a time capsule of the state of the repository when the paper was originally submitted. For those of you who prefer Dockers, check the branch TBA.

## Structure
- **data** : This directory contains different types of data.
  - **evaluation** : Evaluation csv files of OpenSubtitles and Suomi24.
  - **finchat-corpus** : The FinChat corpus itself, with its metadata.
  - **suomi24** : A very small subset of Suomi24 corpus for checking the scripts.
- **environment.yml** : The yml file to create a conda environment.
- **evaluation** : Evaluation scripts to evaluate open-domain Finnish conversational models and the results of human evaluation for baseline models.
  - **automatic-evaluation** : Scripts to calculate automatic metrics for the models.
  - **human-evaluation** : Some scripts to create data for humans to evaluate and the results of those evaluations.
- **models** : Some scripts might assume you have your models in a directory here.
- **training** : Training scripts to train open-domain Finnish conversational models, divided based on architecture.
  -**encoder-decoder** : Encoder-decoder architecture scripts for training and creating the trainig data.
  -**transformer** : Transformer architecture scripts for training and creating the trainig data.
- **website** : Website used during the data collection sessions.
- **LICENSE** : License for all the files expect for FinChat corpus. See finchat-corpus readme for corpus license information.
- **README.md** : Readme-file.

## How to run

First create a conda environment with the environment.yml file, and then activate it. At the time of writing this is done with:
``` 
conda env create -f environment.yml
conda activate FinChat
```
 Then go to the directory of the desired model architecture (at the moment encoder-decoder works with no requirements from user) and use the main.py file for training by giving it a model ID as an input. Use the hyperparameters.py file to choose those.
```
cd training/encoder-decoder
python encoderDecoder_main.py 01
```
To evaluate the model you trained then go to automatic-evaluation directory. The files are once again divided by architecture. So to then evaluate your encoder-decoder model you need to give as arguments the hyperparameters so pytorch can load the model, the text you want to evaluate and the model itself.
```
cd ../../evaluation/automatic-evaluation
python encoderDecoder_evaluate_script.py ../../training/encoder-decoder/encoderDecoder_hyperparameters.py ../../data/evaluation/eval_1k_suomi24_2002.csv ../../models/enc-dec_suomi24/202008131657/360_checkpoint.tar
```

## Citation
K. Leino, J. Leinonen, M. Singh, S. Virpioja and M. Kurimo. "FinChat: Corpus and evaluation setup for Finnish chat conversations on everyday topics." INTERSPEECH. 2020.

## Contact information
See FinChat readme file.

