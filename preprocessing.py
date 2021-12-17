import numpy as np
#Removing HTML
from bs4 import BeautifulSoup
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

import pickle
cv = pickle.load(open('vectorizer.pkl', 'rb'))

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


import string
def remove_punctuations(text):
    text_nopunct = [char for char in text if char not in string.punctuation]
    text_nopunct = "".join(text_nopunct)
    return text_nopunct


#Removing Accented charaacters
import unicodedata
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


#Removing special characters
import re
def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text


import contractions
def expand_contractions(text):
    expanded_words = []    
    for word in text.split():
      expanded_words.append(contractions.fix(word))   
    expanded_text = ' '.join(expanded_words)
    return expanded_text

wn = nltk.WordNetLemmatizer()
def lemmatize_text(text):
    lemmatized_text = " ".join([wn.lemmatize(word) for word in text.split(" ")])
    return lemmatized_text


from nltk.tokenize import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
    return filtered_text

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def bow_transform(text):
    input_X = pd.DataFrame([text]) 
    print("Text is", np.array(input_X).shape)
    return cv.transform(np.array(input_X).ravel())


