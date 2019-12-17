import argparse

import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Embedding, Input


def get_args():
    parser = argparse.ArgumentParser("POS tagging train")
    parser.add_argument("--train_file", help="train dataset file")
    return parser.parse_args()


def get_data(fname):
    df = pd.read_csv(fname, names=['src', 'dst'], sep='\s', engine='python')
    # print(len(lines))
    # print(lines.head())
    # print(lines[pd.isna(lines['dst'])].head())
    src_vocab = sorted(list(set(df['src'])))
    # print(src_vocab)
    # dst_vocab = sorted(list(set([h for line in map(lambda line: line.split("+"), df['dst']) for h in line])))
    dst_vocab = sorted(list(set(df['dst'])))
    # print(dst_vocab)
    return df['src'], df['dst'], src_vocab, dst_vocab


batch_size = 64
epochs = 64
latent_dim = 256
num_samples = 1024 # 학습 데이터 개수

# 문장 벡터화
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

fname = "dataset/train.txt"

# read dataset
lines = open(fname, encoding='utf-8').readlines()
for line in lines[:num_samples]:
    try:
        input_text, target_text = line.split(" ")
        input_texts.append(input_text)
        target_texts.append(target_text)

        # 문자 집합 생성
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    except:
        pass

num_samples = len(input_texts)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', num_samples)
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 학습에 사용할 데이터를 담을 3차원 배열
encoder_input_data = np.zeros(
    (num_samples, max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


# 문자 -> 숫자
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])


# data conversion
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
  for t, char in enumerate(input_text):
    encoder_input_data[i, t, input_token_index[char]] = 1.
  for t, char in enumerate(target_text):
    decoder_input_data[i, t, target_token_index[char]] = 1.
    if t > 0:
      decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# models
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_output, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                    initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          verbose=2)

