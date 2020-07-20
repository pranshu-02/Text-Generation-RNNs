#python3

## Libraries
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
from plotacc import *



## Load Data
data = open('data').read()


## Tokenize Data
corpus = data.lower().split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1


# Create Input Sequences Using Tokens
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


# Pad Sequence To Make Of Equal Length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

## Create Labels And Predictors
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)

## Model: [Embeddings]-> [Bi-Directional LSTM]-> [LSTM]-> [Dense]
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Model Summary
print(model.summary())

## Train Model
history = model.fit(predictors, label, epochs=100, verbose=1)

## Plot Training Accuracy And Training Loss vs Epochs\
accuracy = history.history['accuracy']
loss = history.history['loss']
plotacc(accuracy, loss)

## Uncomment Below Code To Generate Text Using This Model
#### Enter Starting Text In Seed Text And Number of Words Required In Next_Words
##
##seed_text = "You are my only hope"
##next_words = 100
## 
##for _ in range(next_words):
##	token_list = tokenizer.texts_to_sequences([seed_text])[0]
##	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
##	predicted = model.predict_classes(token_list, verbose=0)
##	output_word = ""
##	for word, index in tokenizer.word_index.items():
##		if index == predicted:
##			output_word = word
##			break
##	seed_text += " " + output_word
##print(seed_text)
