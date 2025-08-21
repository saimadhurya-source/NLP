import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

text = "Tokenization without transformers is straightforward with tools like NLTK in Python."

tokens = word_tokenize(text)

print("tokens:",tokens)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization without transformers is straightforward with tools like NLTK"

tokens_transformers = tokenizer(text,return_tensors="pt")

print("\n transformers tokens:",tokens_transformers)

tokens_transformers_list= tokenizer.convert_ids_to_tokens(tokens_transformers['input_ids'][0].numpy().tolist())

print("\n transformers tokens(list):",tokens_transformers_list)

decoded_text= tokenizer.decode(tokens_transformers['input_ids'][0],skip_special_tokens = True)

print("\n decoded text:",decoded_text)
