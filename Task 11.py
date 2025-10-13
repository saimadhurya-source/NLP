import nltk
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def fetch_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch URL: {url} with status code {response.status_code}")
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    raw_text = ' '.join(p.get_text() for p in paragraphs)
    clean_text = re.sub(r'\s+', ' ', raw_text).strip()
    return clean_text

def get_first_sentences(text, num_sentences=3):
    sentences = sent_tokenize(text)
    first_sentences = sentences[:num_sentences]
    first_sentences = [s.rstrip('.').strip() for s in first_sentences]
    return first_sentences


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    text = fetch_text(url)

    if not text:
        print("No text extracted from the URL!")
    else:
        sentences = get_first_sentences(text, 3)
        for sentence in sentences:
            print(sentence)
