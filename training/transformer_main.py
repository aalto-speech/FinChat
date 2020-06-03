#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# 
# Chatbot Tutorial
# ================
# **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
# 
# 

# Preparations
# ------------
# 
# 

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import argparse

from transformer_prep_data import *
from transformer_voc import Voc
from transformer_global_variables import *
from transformer_models import *
from transformer_training import *
from transformer_hyperparameters import *

################################################
######## ALL VARIABLES HERE ####################
################################################

parser = argparse.ArgumentParser(description='Transformer main that ties other modules together')
parser.add_argument('job_name', type=str,
                    help='job id from slurm')

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)


# Define path to new file
inputfile = os.path.join(corpus, source_txt_file)
datafile = os.path.join(corpus, source_csv_file)

save_dir = os.path.join("../models", parent_folder_name, args.job_name)

small_batch_size = 5

#############################################
############ RUNNING THE SCRIPT #############
#############################################

printLines(os.path.join(corpus, source_txt_file))

# Load lines and process conversations
print("\nProcessing corpus...")

# Write new csv file
print("\nWriting newly formatted file...")
createSentencePairsCSV(inputfile, datafile)


# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


# Load/Assemble voc and pairs
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


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


loadFilename = None
#checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    transformer_sd = checkpoint['tra']
    optimizer_sd = checkpoint['opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


ntokens = voc.num_words # the size of vocabulary

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(ntokens, emsize)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
if loadFilename:
    transformer.load_state_dict(encoder_sd)
    
# Use appropriate device
transformer = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, embedding, dropout).to(device)
print('Models built and ready to go!')

# Ensure dropout layers are in train mode
transformer.train()

# Initialize optimizers
print('Building optimizers ...')
optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
if loadFilename:
    optimizer.load_state_dict(optimizer_sd)

# If you have cuda, configure cuda to call
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
    
# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, transformer, optimizer,
           embedding, nlayers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, device)



# Conclusion
# ----------
