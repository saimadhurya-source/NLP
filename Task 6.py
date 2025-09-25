from nltk.util import ngrams
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
def ngram_smoothing(sentence, n):
  tokens = word_tokenize(sentence.lower())
  train_data, padded_sents = padded_everygram_pipeline(n, tokens)
  model = Laplace(n)
  model.fit(train_data, padded_sents)
  return model
sentence = input("Enter a sentence: ")
n = int(input("Enter the value of N for N-grams: "))
model = ngram_smoothing(sentence, n)
context = tuple(sentence.lower().split()[-n+1:])
next_words = model.generate(3, text_seed=context)
print("Next words:", ' '.join(next_words))
