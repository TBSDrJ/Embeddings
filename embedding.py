"""
    Author: TBSDrJ
    Date: Spring 2023
    Purpose: Illustrate word embeddings along the lines of word2vec.
    Reference: https://www.tensorflow.org/text/guide/word_embeddings
    Uses dataset found at: 
        https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    The train and test each had 12 500 entries for each of positive and
        negative reviews.  I decided to combine these to get a single dataset
        with 25 000 entries each of positive and negative and do a random 
        train/validation split.  Note that when combining, some of the 
        filenames for the text files are the same, so one has to be careful
        to avoid overwriting those files in the combination process.
    My combined dataset can be found at:
        https://drive.google.com/file/d/1s-0zOF-FhdUwo2jq5QpIDxR64SFS9pLQ/view?usp=sharing
"""

import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential

BATCH_SIZE = 1024
# If you are using tf version 2.10+, you can avoid using the seed
#   by returning both train and validation sets from the same call
#   to text_dataset_from_directory.
SEED = 42
VALIDATION_SPLIT = 0.3
# Set this to some integer to limit the size of the vocabulary, or None
#   to use every word that is found.
VOCAB_SIZE = 10000
# I believe that this truncates the length of each review to 100 words.
SEQUENCE_LENGTH = 100
# Number of dimensions to capture the meaning of each word.  
#   word2vec used 300 dimensions.
EMBEDDING_DIM = 256

train = utils.text_dataset_from_directory(
    'imdb/combined',
    batch_size = BATCH_SIZE,
    validation_split = VALIDATION_SPLIT,
    subset = 'training',
    seed = SEED,
)
valid = utils.text_dataset_from_directory(
    'imdb/combined',
    batch_size = BATCH_SIZE,
    validation_split = VALIDATION_SPLIT,
    subset = 'validation',
    seed = SEED,
)
train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
valid.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

def clean_text(input_data):
    # Convert all to lower case
    input_data = tf.strings.lower(input_data)
    # Replace all HTML newline tags with spaces
    input_data = tf.strings.regex_replace(input_data, '<br />', ' ')
    # Get rid of everything else that isn't a letter, a space or an apostrophe
    input_data = tf.strings.regex_replace(input_data, "[^a-z'\ ]", '')
    return input_data

vectorize_layer = layers.TextVectorization(
    standardize = clean_text,
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = SEQUENCE_LENGTH,
)

train_text = train.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
vocab = vectorize_layer.get_vocabulary()

model = Sequential(
    [
        vectorize_layer,
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1),
    ]
)

print(model.summary())