import os


# Corpus & Data variables
corpus_name = "opensubs"
corpus = os.path.join("../data", corpus_name)
source_txt_file = "1M_stigs_opensubs.trg"
source_csv_file = "1M_stigs_opensubs.csv"

parent_folder_name = "transformer_opensubs"

# Configure models
model_name = 'transformer_model'
batch_size = 4

emsize = 500 # embedding dimension
nhid = 500 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 10 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

# Configure training/optimization
clip = 0.5
learning_rate = 0.00001
n_iteration = 5600000
print_every = 100
save_every = 800000
