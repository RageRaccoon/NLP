""" NLP Tokenizer Module """

### This module provides a Tokenizer class for text preprocessing tasks such as tokenization, lemmatization, and stopword removal.

import json
import os

from .Lemmatization import Lemmatizer

def read_json(file_path):
    """ Reads a JSON file and returns its content """
    # Get actual folder path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load lemmatization dictionary
    data_path = os.path.join(BASE_DIR, file_path)
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_stopwords(file_path):
    """ Reads a text file and returns its content """
    # Get actual folder path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load lemmatization dictionary
    data_path = os.path.join(BASE_DIR, file_path)
    with open(data_path, 'r', encoding='utf-8') as file:
        data = file.read()
        # Join lines that do not start with '#'
        cleaned_data = "\n".join(line for line in data.splitlines() if not line.startswith("#"))
        # Remove quotes and spaces, then split by commas
        stopwords = [word.strip().strip('"') for word in cleaned_data.split(",") if word.strip()]
    return stopwords




class Tokenizer(Lemmatizer):
    """ Constructor """
    def __init__(self):
        super().__init__()
        data = read_json("Dictionaries/token_utils.json")
        self.delimiter = data["delimiters"]
        self.numbers = data["numbers"]
        self.accents = data["accents"]
        self.stopwords = read_stopwords("Dictionaries/stopwords.data")
        self.unique_words = set()

    """ Methods """
    # Verifies if the word is only numbers or alphanumeric
    def verify_word(self, text:str) -> str:
        is_only_number = True
        word = ""
        for char in text:
            if char not in self.numbers:
                is_only_number = False
                break 

        if is_only_number:
            word = text
        else:
            # Keep alphabetic characters, remove only numbers from mixed words
            for char in text:
                if char.isalpha():
                    word += char
        return word
    
    # Converts all characters in the token to lowercase
    def to_lowercase(self, token:list) -> list:
        for i in range(len(token)):
            for c in token[i]:
                if (c >= 'A') and (c <= 'Z'):
                    token[i] = token[i].replace(c, chr(ord(c) + 32))
        return token
    
    # Delete stopwords from the token
    def remove_stopwords(self, token:list) -> list:
        return [word for word in token if word not in self.stopwords]

    # Delete accent marks from the token
    def remove_accents(self, token:list) -> list:
        new_token = []
        for word in token:
            new_word = ""
            for char in word:
                if char in self.accents:
                    new_word += self.accents[char]
                else:
                    new_word += char
            new_token.append(new_word)
        return new_token
    
    # Define the function to save the vocabuylary of unique words
    def save_unique_words(self, token:list):
        for word in token:
            self.unique_words.add(word)
            
    # Get the saved vocabulary of unique words as a list
    def get_unique_words(self):
        return list(self.unique_words)
        
    
    # Tokenizes the input text
    def tokenize(self, text: str) -> list:              
        token = []
        n = len(text)
        
        i = 0
        j = i
        
        while i <= n - 1:
            if (text[i] in self.delimiter) and (text[j] in self.delimiter):
                j += 1
            elif (text[i] in self.delimiter):
                word_verified = self.verify_word(text[j:i])
                if word_verified: 
                    token.append(word_verified)
                j = i + 1
            i += 1

        # Handle the last word if the text doesn't end with a delimiter
        if j < n:
            word_verified = self.verify_word(text[j:n])
            if word_verified:
                token.append(word_verified)

        token = self.to_lowercase(token)
        
        token = self.remove_accents(token)
        
        token = self.remove_stopwords(token)

        token = self.lemmatize(token)
        
        self.save_unique_words(token)

        return token