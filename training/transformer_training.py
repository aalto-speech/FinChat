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

from transformer_prep_data import *
from transformer_voc import Voc
from transformer_global_variables import *


criterion = nn.NLLLoss(ignore_index=0)


def train(input_variable, lengths, target_variable, mask, max_target_len, transformer, embedding,
          optimizer, ntokens, batch_size, clip, device, max_length=MAX_LENGTH):

    # Zero gradients
    optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Forward pass through encoder
    outputs = transformer(input_variable)
    
    loss = criterion(outputs.view(-1, ntokens), target_variable.view(-1))
    
#    for t in range(max_target_len):
#        # Calculate and accumulate loss
#        mask_loss, nTotal = maskNLLLoss(outputs[t], target_variable[t], mask[t])
#        loss += mask_loss
#        print_losses.append(mask_loss.item() * nTotal)
#        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(transformer.parameters(), clip)

    # Adjust model weights
    optimizer.step()

    return loss.item()


# In[18]:


def trainIters(model_name, voc, pairs, transformer, optimizer, embedding, nlayers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename, device):

    ntokens = voc.num_words

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, transformer,
                     embedding, optimizer, ntokens, batch_size, clip, device)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'tra': transformer.state_dict(),
                'opt': optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
