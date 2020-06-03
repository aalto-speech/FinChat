from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import argparse

from encoderDecoder_global_variables import SEED

parser = argparse.ArgumentParser(description='Encoder-Decoder create eval set from file')
parser.add_argument('source_file_name', type=str,
                    help='name of the large eval file')
parser.add_argument('target_file_name', type=str,
                    help='name of the smaller shuffled eval file')
parser.add_argument('how_many', type=int,
                    help='how many lines to new dataset')

args = parser.parse_args()

random.seed(SEED)
min_length_characters = 10
N = 10

# Extracts pairs of sentences from conversations
def createSentencePairsList(inputfilename):
    qa_pairs = []
    inputLine = ""
    targetLine = "hyvää uutta vuotta !"
    with open(inputfilename, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            inputLine = targetLine
            targetLine = line.strip()            
            # Filter wrong samples (if one of the lists is empty)
            if len(inputLine) > min_length_characters and len(targetLine) > min_length_characters:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def create_N_choose_k_file_from_list(qa_pairs, output_csv_file_name, N):
    # TODO should I only pick long enough sentences?
    with open(output_csv_file_name, 'w', encoding='utf-8') as output_file:
        lines_count = len(qa_pairs)
        assert lines_count >= N + 2, "Not enough lines, eg. options for fake"
        output_file.write("TEXT¤CHOICE_SENTENCES\n")

        for i in range(lines_count - 1):
            bad_indices = []
            bad_indices.append(i)
            answers = []
            question = qa_pairs[i][0].strip()
            true_answer = qa_pairs[i][1].strip()
            answers.append(true_answer)

            for _ in range(N - 1):
                fake_answer = ""
                while fake_answer == "":
                    fake_answer_index = random.choice([x for x in range(lines_count) if x not in bad_indices])
                    fake_answer = qa_pairs[fake_answer_index][1].strip()
                    # I don't think it can be true_answer, but can be question
                    if fake_answer == question or fake_answer == true_answer:
                        fake_answer = ""

                answers.append(fake_answer)
                bad_indices.append(fake_answer_index)

            line_to_write = '¤'.join([question, '|'.join(answers)]) + '\n'
            output_file.write(line_to_write)


qa_pairs = createSentencePairsList(args.source_file_name)
random.shuffle(qa_pairs)

create_N_choose_k_file_from_list(qa_pairs[:args.how_many + 1], args.target_file_name, N)
