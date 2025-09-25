import string
import random
import nltk
from nltk.corpus import stopwords, reuters
from collections import Counter, defaultdict
from nltk import FreqDist, ngrams

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('reuters')
nltk.download('punkt_tab')

sents = reuters.sents()
stop_words = set(stopwords.words('english'))
# Create a translation table to remove punctuation
translator = str.maketrans('', '', string.punctuation)

unigrams = []
bigrams = []
trigrams = []

for sentence in sents:
    # Convert to lowercase and remove punctuation
    cleaned_sentence = [word.lower().translate(translator) for word in sentence if word.isalpha() or word in string.punctuation]
    # Remove empty strings resulting from punctuation removal
    cleaned_sentence = [word for word in cleaned_sentence if word]
    # Remove stopwords
    cleaned_sentence = [word for word in cleaned_sentence if word not in stop_words]

    if len(cleaned_sentence) > 0: # Process only if the sentence is not empty after cleaning
        unigrams.extend(cleaned_sentence)
        bigrams.extend(list(ngrams(cleaned_sentence, 2, pad_left=True, pad_right=True)))
        trigrams.extend(list(ngrams(cleaned_sentence, 3, pad_left=True, pad_right=True)))

freq_uni = FreqDist(unigrams)
freq_bi = FreqDist(bigrams)
freq_tri = FreqDist(trigrams)

d = defaultdict(Counter)
# Only add trigrams where all elements are not None
for a, b, c in trigrams:
    if a is not None and b is not None and c is not None:
        d[(a, b)][c] += 1 # Increment count for each valid trigram

def pick_word(counter):
    "choose a random element based on frequency"
    return random.choices(list(counter.keys()), weights=list(counter.values()), k=1)[0]

prefix = "he", "is"
print(" ".join(prefix))
s = " ".join(prefix)

for i in range(19):
    if prefix in d:
        suffix = pick_word(d[prefix])
        s = s + ' ' + suffix
        print(s)
        prefix = prefix[1], suffix
    else:
        print(f"Prefix {prefix} not found in dictionary. Stopping text generation.")
        break
