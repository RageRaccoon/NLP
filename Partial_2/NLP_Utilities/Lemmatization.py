""" NLP Lemmatization Module """

### This module provides a Lemmatizer class for performing lemmatization on tokens using a predefined dictionary.

import json
import os

class Lemmatizer:
    """ Constructor """
    def __init__(self):
        # Get actual folder path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Load lemmatization dictionary
        json_path = os.path.join(BASE_DIR, "Dictionaries", "lemmatization_utils.json")
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.lemmatization_dict = data["lemma_words"]

    """ Methods """
    # Add the new normalized word to the dictionary if not present
    def add_to_dictionary(self, word: str, lemma: str):
        if word not in self.lemmatization_dict:
            # Get actual folder path
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            # Load lemmatization dictionary
            json_path = os.path.join(BASE_DIR, "Dictionaries", "lemmatization_utils.json")
            with open(json_path, 'r+', encoding='utf-8') as file:
                data = json.load(file)
                data["lemma_words"][word] = lemma
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

    # Normalization function
    def normalize(self, word: str) -> str:
        # Plural to singular
        if word.endswith("es") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        
        # For regular verbs in infinitive form
        if word.endswith("ando"):
            word = word[:-4] + "ar"
        elif word.endswith("iendo"):
            word = word[:-5] + "er"
            
        if word.endswith("ado"):
            word = word[:-3] + "ar"
        elif word.endswith("ido"):
            word = word[:-3] + "er"
        
        return word
        

    # Lemmatization function
    def lemmatize(self, token:list) -> list:
        new_token = []
        
        for word in token:
            if word in self.lemmatization_dict:
                new_token.append(self.lemmatization_dict[word])
            else:
                normalized_word = self.normalize(word)
                new_token.append(normalized_word)
                # self.add_to_dictionary(word, normalized_word)

        return new_token