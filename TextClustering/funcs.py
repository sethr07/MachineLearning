#!/usr/bin/python3

"""
Utilities file
"""

import string 
import nltk 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

test = ["My name is Rahul. How are you? What's your, name?", "heyy im rahul."]

def vector(list):
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(list)
    return X


def clean(msg):
    nonPunc = "".join([char.lower() for char in msg if char not in string.punctuation])
    stemmer = PorterStemmer()
    cleaned = stemmer.stem(msg)
    return cleaned


#cleaned = clean(test)
#print(cleaned)