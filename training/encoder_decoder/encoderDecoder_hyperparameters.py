__author__ = "Original code by Matthew Inkawhich <https://github.com/MatthewInkawhich>, modified by Juho Leinonen"
__copyright__ = "BSD 3-Clause license, 2017, Pytorch contributors"
# Contains the values for the hyperparameters used to train the model (varies between models).

import os


# Corpus & Data variables 
corpus_name = "suomi24"
corpus_location = os.path.join("../../data", corpus_name)
source_txt_file = "1k_suomi_24_morfs.txt"
source_csv_file = "1k_suomi_24_morfs.csv"

parent_folder_name = "enc-dec_suomi24"

# Configure models
model_name = 'encoderDecoder'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 4
decoder_n_layers = 4
dropout = 0.2
batch_size = 8

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.95
learning_rate = 0.00001
decoder_learning_ratio = 5.0
n_iteration = 5600
print_every = 100
save_every = 5600
