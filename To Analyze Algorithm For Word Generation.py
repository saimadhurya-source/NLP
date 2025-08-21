import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import spacy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download("punkt")
nltk.download("punkt_tab")
nlp = spacy.load("en_core_web_sm")

corpus ="One disadvantage of using 'Best Of' samping is that it may lead to limited exploration of the model's nowledge and creativity. By focusing on the most probable next words, the model might generate responses that are safe and conventional, potentially missing out on more diverse and innovative outputs. The lack of exploration could esult in repetitive or less imaginative responses, especially in situations where novel and unconventional ideas are desired.To address this limitation, other sampling strategies like temperature-based sampling or top-p (nucleus) sampling can be employed to introduce more randomness and encourage the model to explore a broader range of possibilities. However, it's essential to carefully balance exploration and exploitation based on the specific requirements of the task or application."

tokens = word_tokenize(corpus)
lemmatized_tokens = [token.lemma_ for token in nlp(corpus)]
all_tokens = tokens + lemmatized_tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tokens)
total_words = len(tokenizer.word_index) + 1
input_sequences = []

for line in all_tokens:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = np.array(y)
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=1)
