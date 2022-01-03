import torch
import pandas as pd
import fasttext
import math
import numpy as np

def init_embedding(fasttext_embed_file):

    """
    Initializes fast text word embedding matrix
    """
    
    df_embedding = pd.read_csv(fasttext_embed_file, sep=" ", quoting=3, header=None, index_col=0,skiprows=1)

    embedding_matrix = df_embedding.to_numpy()

    vocab = []

    for word in list(df_embedding.index):
        vocab.append(str(word))

    vocab_size , vocab_dim = embedding_matrix.shape

    word2idx = {w: idx for (idx, w) in enumerate(vocab)}

    idx2word = {idx: w for (idx, w) in enumerate(vocab)}

    return embedding_matrix, vocab_size, vocab_dim, word2idx



def tokenized_tensor(data, word2idx):
    
    """
    Returns tokenized tensor vector for input text=
    """
    output_tokenized = []

    for sentence in data:
        output = []
        tokenized = fasttext.tokenize(sentence)

        for word in tokenized:
            if word in word2idx:
                id = word2idx[word]
                output.append(id)
            else:
                word2idx[word] = len(word2idx)
                id = word2idx[word]
                output.append(id)

        output = torch.tensor(output)

        output_tokenized.append(output)

    return output_tokenized, word2idx


def get_data_sequences(train_dir, valid_dir, test_dir, word2idx):
    
    """
    Returns tokenized tensor sequences for the dataset
    """
    train_df = pd.read_csv(train_dir)

    val_df = pd.read_csv(valid_dir)

    test_df = pd.read_csv(test_dir)

    train_data =  train_df['tweets'].values
    train_labels = train_df['labels'].values

    val_data =  val_df['tweets'].values
    val_labels = val_df['labels'].values

    test_data =  test_df['tweets'].values
    test_labels = test_df['labels'].values

    train_tokenized_sequences, word2idx = tokenized_tensor(train_data, word2idx)

    test_tokenized_seqences, word2idx = tokenized_tensor(test_data, word2idx)

    val_tokenized_sequences, word2idx = tokenized_tensor(val_data, word2idx)

    return train_tokenized_sequences, train_labels, test_tokenized_seqences, test_labels, val_tokenized_sequences, val_labels, word2idx


## Create final embedding matrix

def create_embedding(embedding_matrix, word2idx, vocab_size, vocab_dim):
    
    """
    Returns complete fasttext embedding matrix
    """

    word2idx['<PAD>'] = len(word2idx)

    random_init = torch.nn.Parameter(torch.Tensor( (len(word2idx) - vocab_size), vocab_dim))
    torch.nn.init.kaiming_uniform_(random_init, a=math.sqrt(5))


    new_matrix = np.zeros( (len(word2idx), vocab_dim) )

    new_matrix[:vocab_size, :] = embedding_matrix

    embedding_matrix = new_matrix

    embedding_matrix[vocab_size:, :] = random_init.detach().numpy()

    return embedding_matrix, word2idx