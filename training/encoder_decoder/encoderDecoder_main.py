from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Original code by Matthew Inkawhich <https://github.com/MatthewInkawhich>, modified by Juho Leinonen"
__copyright__ = "BSD 3-Clause license, 2017, Pytorch contributors"
# The script that prepares the data and trains the models.

# coding: utf-8
# 
# Chatbot Tutorial
# ================
# **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
# 

import argparse
import random
import torch

from encoderDecoder_prep_data import printLines
from encoderDecoder_prep_data import createSentencePairsCSV
from encoderDecoder_prep_data import loadPrepareData
from encoderDecoder_prep_data import batch2TrainData
from encoderDecoder_prep_data import trimRareWords

from encoderDecoder_global_variables import MIN_COUNT, SEED

from encoderDecoder_models import EncoderRNN
from encoderDecoder_models import LuongAttnDecoderRNN
from encoderDecoder_training import trainIters

from encoderDecoder_hyperparameters import *

################################################
######## ALL VARIABLES HERE ####################
################################################

parser = argparse.ArgumentParser(description='Encoder-Decoder main that ties other modules together')
parser.add_argument('job_name', type=str,
                    help='job id from slurm')

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

# Define path to new file
inputfile = os.path.join(corpus_location, source_txt_file)
datafile = os.path.join(corpus_location, source_csv_file)
save_dir = os.path.join("../../models", parent_folder_name, args.job_name)

small_batch_size = 5

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

###############################################
######### RUNNING THE SCRIPT ##################
###############################################

printLines(os.path.join(corpus_location, source_txt_file))

# Load & Preprocess Data
# ----------------------
print("\nProcessing corpus...")

# Write new csv file
print("\nWriting newly formatted file...")
createSentencePairsCSV(inputfile, datafile)

# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


# Print some pairs to validate
voc, pairs = loadPrepareData(corpus_location, corpus_name, datafile, save_dir)
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

# Load and trim data
# ~~~~~~~~~~~~~~~~~~

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


# Example for validation
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = torch.nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Run Training
# ~~~~~~~~~~~~
# 

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
    
# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, teacher_forcing_ratio, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, device)





# Conclusion
# ----------
