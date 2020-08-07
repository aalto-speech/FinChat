__author__ = "Original code by Matthew Inkawhich <https://github.com/MatthewInkawhich>, modified by Juho Leinonen"
__copyright__ = "BSD 3-Clause license, 2017, Pytorch contributors"
# Contains the values for the global variables used in other files.

SEED = 1337
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token

EOS_token = 2  # End-of-sentence token
EOT_token = 3  # End-of-turn token

MAX_LENGTH = 30  # Maximum sentence length to consider
MIN_COUNT = 2    # Minimum word count threshold for trimming
