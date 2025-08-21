import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')

def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize_document(document):
    tokens = word_tokenize(document)
    return [word.lower() for word in tokens if word.isalpha()]

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def find_morphology(tokens):
    fdist = FreqDist(tokens)
    return fdist

document_path = '/content/drive/MyDrive/NLP/task4_dataset.txt'

document = load_document(document_path)
tokens = tokenize_document(document)
tokens_without_stopwords = remove_stopwords(tokens)
morphology = find_morphology(tokens_without_stopwords)
print("Morphology of the document:")

for word, frequency in morphology.items():
    print(f"{word}: {frequency}")
