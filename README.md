# Embeddings

A quick demonstration of how to build an ML model for how to encode word embeddings in a vector space.

I used [this tutorial](https://www.tensorflow.org/text/guide/word_embeddings) extensively.

The dataset is a large dataset of [IMDB reviews](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), 
classifying the review as 'positive' or 'negative.' 

I [reorganized the dataset](https://drive.google.com/file/d/1s-0zOF-FhdUwo2jq5QpIDxR64SFS9pLQ/view?usp=sharing) 
to combine the train and test datasets because they had a 50/50 split 
which seemed excessive on the test side, so I built a combined dataset with all reviews, and 
then did a random 70/30 split in the code.  This changes the problem a bit because the vocabulary 
that my dataset builds differs from the vocabulary used in the original problem, so if you want to 
solve the original problem, use the original Stanford dataset unmodified.  In my combined dataset,
reviews that came from the train dataset are in files that start with '0_' and reviews that came 
from the test dataset are in files that start with '1_'.
