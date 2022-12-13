import os, json
import numpy as np
import pandas as pd
import re
import h5py

BASE_DIR = '/content/drive/Shareddrives/CS260-Project'

def load_lyric_data(num_topics, base_dir=BASE_DIR, max_train=None):
    '''
    HOW TO USE:
        num_topics: specify how many topics to use for training
            options - 6, 10, 20, 30, 40
        base_dir: base directory to pull all files from
        max_train: decide how much data to subsample
    '''
    
    # print the base directory
    print('base dir ', base_dir)
    
    # initialize
    data = {}
    topics = {}
    
    # fill in the initial data
    train_file = os.path.join(base_dir, f"data/train/big-lda-train-{num_topics}.csv")
    val_file = os.path.join(base_dir, f"data/val/big-lda-val-{num_topics}.csv")
    pd_train_data = pd.read_csv(train_file)
    pd_val_data = pd.read_csv(val_file)
    
    for k, v in pd_train_data.items():
      stri = f'train_{k}'
      data[stri] = np.asarray(v)
    
    for k, v in pd_val_data.items():
      stri = f'val_{k}'
      data[stri] = np.asarray(v)
    
    # open data files
    lyric_file = os.path.join(base_dir, f'src/vanilla-rnn/h5/big-lda-{num_topics}.h5')
    with h5py.File(lyric_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)
      
    # Fill in the tokenizer
    dict_file = os.path.join(base_dir, "src/vanilla-rnn/word-to-idx.json")
    with open(dict_file, "r") as f:
      dict_data = json.load(f)
      for k, v in dict_data.items():
        data[k] = v
    
    # Build the feature mappings here [artist, topic_id]
    # this builds the artist to idx mapping, so we don't need to change this
    dict_file = os.path.join(base_dir, "src/vanilla-rnn/artist-to-idx.json")
    with open(dict_file, "r") as f:
      dict_data = json.load(f)
      for k, v in dict_data.items():
        data[k] = v

    # we also need to build a feature mapping space for each word representation
    # for the topic, and here is where we decide how to embed it
    dict_file = os.path.join(base_dir, f"src/vanilla-rnn/train_updated/lda-{num_topics}-topic-embeddings.json")
    with open(dict_file, "r") as f:
      dict_data = json.load(f)
      for k, v in dict_data.items():
        topics[k] = v
    
    artists_tokenized = [[data['artist_to_index'][x]] for x in data['train_artist']]
    val_artists_tokenized = [[data['artist_to_index'][x]] for x in data['val_artist']] 
    train_topic_embeddings = [topics[str(x)] for x in data['train_topic_id']]
    val_topic_embeddings = [topics[str(x)] for x in data['val_topic_id']]
    
    # [[x, x, x,...],[],[]...]
    # [[artist_id, x, x, x,...][][]...] (L, 50*20=T)
    # np.concatenate? (L, T+1)
    # artists_tokenized (L, 1)
    data['train_features'] = np.concatenate((artists_tokenized, train_topic_embeddings), axis=1)
    data['val_features'] =np.concatenate((val_artists_tokenized, val_topic_embeddings), axis=1)

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data["train_lyric"].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data["train_lyric"] = data["train_lyric"][mask]
    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]] # sounds like our tokenizer/embedding
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_lyric_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_lyric" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    lyrics = data["%s_lyric" % split][mask]
    features = data["%s_features" % split][mask]
    return lyrics, features


def refine_lyrics(lyric, max_len):
    lyrics_list = lyric.lower()
    lyrics_list = re.sub(r'[-]', ' ', lyrics_list)
    lyrics_list = re.sub(r'[^\s+a-z+]+', '', lyrics_list)
    return lyrics_list.split()[:max_len-2]
    

def write_lyric_data(train_file, val_file, write_file, base_dir='/content/drive/Shareddrives/CS260-Project', max_lyric_size=512):
    # print the base directory
    print('base dir ', base_dir)
    
    # initialize
    data = {}
    
    # open and read training data
    print('(1/6) Extracting training data...')
    train_file = os.path.join(base_dir, train_file)
    pd_train_data = pd.read_csv(train_file)
    
    for k, v in pd_train_data.items():
      stri = f'train_{k}'
      data[stri] = np.asarray(v)
      
    # open and read validation data
    print('(2/6) Extracting validation data...')
    val_file = os.path.join(base_dir, val_file)
    pd_val_data = pd.read_csv(val_file)
    
    for k, v in pd_val_data.items():
      stri = f'val_{k}'
      data[stri] = np.asarray(v)
      
    # Fill in the tokenizer
    print('(3/6) Extracting tokenizer...')
    dict_file = os.path.join(base_dir, "src/vanilla-rnn/word-to-idx.json")
    with open(dict_file, "r") as f:
      dict_data = json.load(f)
      for k, v in dict_data.items():
        data[k] = v

    # Reformat lyrics from word to idx
    lyric_embed = []
    lyric_embed_val = []
    max_len = max_lyric_size
    
    print('(4/6) Tokenizing training lyrics...')
    for lyric in data['train_lyric']:
      fragment = [data['word_to_index']['<START>']]
      if type(lyric) == str:
        fragment += [data['word_to_index'][x] if x in data['word_to_index'] else data['word_to_index']['<UNK>'] for x in refine_lyrics(lyric, max_len)]
      fragment.append(data['word_to_index']['<END>'])
      lyric_embed.append(np.pad(fragment, (0, max_len-len(fragment)), 'constant', constant_values=0))
    
    print('(5/6) Tokenizing validation lyrics...')
    for lyric in data['val_lyric']:
      fragment = [data['word_to_index']['<START>']]
      if type(lyric) == str:
        fragment += [data['word_to_index'][x] if x in data['word_to_index'] else data['word_to_index']['<UNK>'] for x in refine_lyrics(lyric, max_len)]
      fragment.append(data['word_to_index']['<END>'])
      lyric_embed_val.append(np.pad(fragment, (0, max_len-len(fragment)), 'constant', constant_values=0))
      
    data_train = np.asarray(lyric_embed)
    data_val = np.asarray(lyric_embed_val)

    # store as h5 files
    print(f'(6/6) Storing data into h5 file {write_file}...')
    h5_train_file = os.path.join(base_dir, write_file)
    hf = h5py.File(h5_train_file, 'w')

    hf.create_dataset('train_lyric', data=data_train)
    hf.create_dataset('val_lyric', data=data_val)
    hf.close()
    
    print('Done!')