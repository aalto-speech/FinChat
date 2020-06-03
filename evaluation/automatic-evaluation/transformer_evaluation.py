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
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf

from spacy.lang.fi import Finnish

from transformer_global_variables import *
from transformer_prep_data import *


# Define Evaluation
# -----------------
#
# Greedy decoding
# ~~~~~~~~~~~~~~~
#



# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
def generate(transformer, voc, prepared_question, device, max_length=MAX_LENGTH):
    prepared_question_tokens = prepared_question.split()
    input = torch.Tensor([[1]]).long().to(device) # SOS token
    for morph in prepared_question_tokens:
        word_idx = voc.word2index[morph]
        word_tensor = torch.Tensor([[word_idx]]).long().to(device)
        input = torch.cat([input, word_tensor], 0)
    word_tensor = torch.Tensor([[3]]).long().to(device)
    input = torch.cat([input, word_tensor], 0)

    temperature = 1.0
    hypothesis = []

    with torch.no_grad():  # no tracking history
        for i in range(max_length):
            output = transformer(input, False)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)

            word = voc.index2word[word_idx.item()]
            hypothesis.append(word)

    return hypothesis


def calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, transformer, voc, criterion, embedding, device, batch_size,
                   max_length=MAX_LENGTH):
    # Set device options
    ntokens = voc.num_words  # the size of vocabulary
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Forward pass through encoder
    outputs = transformer(input_variable)

    loss = criterion(outputs.view(-1, ntokens), target_variable.view(-1))

    return loss.item()


def prepare_sentence(s, voc):
    s_norm = normalizeString(s)
    s_morfs = s_norm.split()
    return_morfs = []
    for morf in s_morfs:
        if morf not in voc.word2index:
            continue
        else:
            return_morfs.append(morf)
    return " ".join(return_morfs)


def morfenize_fi(text, morfessorModel, spacy_fi):
    text = text.replace(" <MS> ", " . ")
    text = text.lower()
    tokens = [tok.text for tok in spacy_fi.tokenizer(text)]
    sentenceAsMorfs = []
    for token in tokens:
        morfs, _ = morfessorModel.viterbi_segment(token)
        if len(morfs) == 1:
            sentenceAsMorfs.append(morfs[0])
        else:
            sentenceAsMorfs.append(morfs[0] + "+")
            for morf in morfs[1:-1]:
                sentenceAsMorfs.append("+" + morf + "+")
            sentenceAsMorfs.append("+" + morfs[-1])
    return " ".join(sentenceAsMorfs)


def morf_list_to_word_list(sentence):
    word_sentence = " ".join(sentence)
    word_sentence = word_sentence.replace("+ +","").replace(" +", "").replace("+ ", "")
    word_sentence = word_sentence.split()
    return word_sentence


def calculate_evaluation_metrics(eval_file_name, voc, transformer, embedding, N, k, delimiter, device, skip_indices=[], print_indices=[], morfessor=None):
    criterion = nn.NLLLoss(ignore_index=0)
    spacy_fi = Finnish()

    true_first = 0
    true_top_k = 0
    corpus_hypothesis = []
    corpus_references = []
    true_answer_losses = []
    hypotheses_for_humans = []

    df = pd.read_csv(eval_file_name, sep=delimiter, engine='python')
    for index, row in df.iterrows():
        if index in skip_indices:
            continue

        question = row['TEXT'].strip()  # TODO what if question or answer is zero, make sure it is not in create file?
        if morfessor:
            question = morfenize_fi(question, morfessor, spacy_fi)

        answers = row['CHOICE_SENTENCES'].split('|')
        assert len(answers) >= N, "CSV file does not have enough choices for value of given N"
        answers = answers[:10]
        assert N >= k, "N is not larger than or equal k"

        losses = []
        prepared_question = prepare_sentence(question, voc)

        first_answer = True
        for answer in answers:
            answer = answer.strip()
            if morfessor:
                answer = morfenize_fi(answer, morfessor, spacy_fi)

            prepared_answer = prepare_sentence(answer, voc)

            # Following gets the length for character normalized perplexity, and saves ref and hyp for BLEU
            if first_answer:

                correct_answer_length_char = max(len(prepared_answer), 1)
                correct_answer_length_tokens = max(len(prepared_answer.split(' ')), 1)

                # Calculate loss of question so it can be subtracted
                question_to_calculate = prepared_question + " EOT"
                evaluation_batch = [batch2TrainData(voc, [question_to_calculate.strip()])]
                input_variable, lengths, target_variable, mask, max_target_len = evaluation_batch[0]

                question_loss = calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, transformer,
                                      voc, criterion, embedding, device, 1)
                normalized_question_loss = question_loss * len(question_to_calculate.strip().split(' '))

                # Had some problem with indexing so this is done twice for every row
                sentence_to_test = prepared_question + " EOT " + prepared_answer
                evaluation_batch = [batch2TrainData(voc, [sentence_to_test.strip()])]
                input_variable, lengths, target_variable, mask, max_target_len = evaluation_batch[0]

                loss = calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, transformer,
                                      voc, criterion, embedding, device, 1)
                normalized_loss = loss * len(sentence_to_test.strip().split(' '))
                answer_loss = (normalized_loss - normalized_question_loss) / correct_answer_length_tokens
                true_answer_losses.append([answer_loss, correct_answer_length_char, correct_answer_length_tokens])
                first_answer = False

                # Next is for BLEU
                hypothesis = generate(transformer, voc, prepared_question, device, max_length=MAX_LENGTH)
                try:
                    first_EOS_index = hypothesis.index(voc.index2word[EOS_token])
                except ValueError:
                    first_EOS_index = MAX_LENGTH  # Generated hypothesis has 50 tokens, none is EOS, so is added as 51th.
                hypothesis = hypothesis[:first_EOS_index]
                corpus_hypothesis.append(hypothesis)
                if index in print_indices:
                    hypothesis_string = " ".join(morf_list_to_word_list(hypothesis))
                    hypotheses_for_humans.append([str(index), row['TEXT'].strip(), hypothesis_string])

                answer_in_tokens = answer.split()
                corpus_references.append(answer_in_tokens)


            sentence_to_test = prepared_question + " EOT " + prepared_answer
            evaluation_batch = [batch2TrainData(voc, [sentence_to_test.strip()])]
            input_variable, lengths, target_variable, mask, max_target_len = evaluation_batch[0]

            loss = calculate_loss(input_variable, lengths, target_variable, mask, max_target_len, transformer, voc,
                                  criterion, embedding, device, 1)
            normalized_loss = loss * len(sentence_to_test.strip().split(' '))
            answer_length_tokens = max(len(prepared_answer.split(' ')), 1)
            answer_loss = (normalized_loss - normalized_question_loss) / answer_length_tokens
            losses.append(answer_loss)
        if np.argmin(np.asarray(losses)) == 0:
            true_first += 1
        if 0 in np.asarray(losses).argsort()[:k]:
            true_top_k += 1

    fraction_of_correct_firsts = true_first / len(true_answer_losses)
    franction_of_N_choose_k = true_top_k / len(true_answer_losses)

    np_true_answer_losses = np.asarray(true_answer_losses)
    #perplexity = np.exp(np.mean(np_true_answer_losses[:,0]))
    cross_entropy = np.mean(np_true_answer_losses[:,0])

    token_to_character_modifier = np_true_answer_losses[:,2] / np_true_answer_losses[:,1]
    #char_perplexity = np.exp(np.mean(np_true_answer_losses[:,0] * token_to_character_modifier))
    char_cross_entropy = np.mean(np_true_answer_losses[:,0] * token_to_character_modifier)

    bleu_morf = corpus_bleu(corpus_references, corpus_hypothesis)
    chrf_morf = corpus_chrf(corpus_references, corpus_hypothesis)
    
    corpus_references_word = [morf_list_to_word_list(sentence) for sentence in corpus_references]
    corpus_hypothesis_word = [morf_list_to_word_list(sentence) for sentence in corpus_hypothesis]
    print(corpus_hypothesis_word)
    print("FOR HUMANS")
    for answer_for_human in hypotheses_for_humans:
        print(" --- ".join(answer_for_human))

    bleu_word = corpus_bleu(corpus_references_word, corpus_hypothesis_word)
    chrf_word = corpus_chrf(corpus_references_word, corpus_hypothesis_word)

    return fraction_of_correct_firsts, franction_of_N_choose_k, cross_entropy, char_cross_entropy, bleu_word, bleu_morf, chrf_word, chrf_morf


def create_N_choose_k_file(source_txt_file_name, output_csv_file_name, N):
    # TODO should I only pick long enough sentences?
    with open(source_txt_file_name, 'r', encoding='utf-8') as source_file, \
            open(output_csv_file_name, 'w', encoding='utf-8') as output_file:
        eval_lines = source_file.readlines()
        lines_count = len(eval_lines)
        assert lines_count >= N + 2, "Not enough lines, eg. options for fake"
        output_file.write("TEXT¤CHOICE_SENTENCES\n")

        for i in range(lines_count - 1):
            bad_indices = []
            bad_indices.append(i)
            bad_indices.append(i + 1)
            answers = []
            question = eval_lines[i].strip()
            true_answer = eval_lines[i + 1].strip()
            if not (len(question) > 10 and len(true_answer) > 10):
                continue
            answers.append(true_answer)

            for _ in range(N - 1):
                fake_answer = random.choice([x for x in range(lines_count) if x not in bad_indices])
                answers.append(eval_lines[fake_answer].strip())
                bad_indices.append(fake_answer)

            line_to_write = '¤'.join([question, '|'.join(answers)]) + '\n'
            output_file.write(line_to_write)
