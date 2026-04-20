from tensorflow.keras.preprocessing.text import Tokenizer

with open("data/questions.txt", "r") as f:
    questions = f.readlines()

for q in questions :
    print(q.strip())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
print(tokenizer.word_index)

sequences = []

for q in questions:
    token_list = tokenizer.texts_to_sequences([q])[0]
    for i in range(1, len(token_list)):
        sequences.append(token_list[:i+1])

print("Total sequences:", len(sequences))
print("Example sequence:", sequences[0])
print("Example sequence:", sequences[1])
print("Example sequence:", sequences[2])

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = max(len(s) for s in sequences)
print("Longest sequence:", max_len)

sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print("After padding:", sequences[0])
print("After padding:", sequences[1])

import numpy as np

X = sequences[:, :-1]
y = sequences[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Example X:", X[0])
print("Example y:", y[0])

from tensorflow.keras.utils import to_categorical

vocab_size = len(tokenizer.word_index) + 1
print("Total unique words:", vocab_size)

y = to_categorical(y, num_classes=vocab_size)
print("y shape after one hot:", y.shape)