"""NLP Bag of Words Model Implementation"""

### This module provides a BagOfWords class for creating a bag-of-words representation of text data.

import pandas as pd
from .Tokenizer import Tokenizer

class BagOfWords(Tokenizer):
    """ Constructor """
    def __init__(self, ):
        super().__init__()
        self.documents = []
        self.tittles = []   
        self.labels = []
        self.tokens = []
        self.vocabularies = []
       

    def fit_transform(self, docs:list, tittles:list=None, labels:list=None):
        self.documents = docs
        self.tittles = tittles if tittles is not None else [f"Document {i+1}" for i in range(len(docs))]
        self.labels = labels if labels is not None else [f"Doc_{i+1}" for i in range(len(docs))]
        
        self.tokens = []
        self.vocabularies = []
        
    
    """ Methods """
    # Creates a bag-of-words representation from a list of texts
    def compute_bow(self) -> pd.DataFrame:
        # Tokenize each document and build vocabulary
        for doc in self.documents:
            doc_tokens = self.tokenize(doc)
            self.tokens.append(doc_tokens)
            self.vocabularies.append(self.get_unique_words())
            
        unique_vocab = set().union(*self.vocabularies)
        unique_vocab = sorted(list(unique_vocab))
        
        bow_data = []
        for tokens in self.tokens:
            word_count = {word: 0 for word in unique_vocab}
            for word in tokens:
                if word in word_count:
                    word_count[word] += 1
            bow_data.append(word_count)
            
        bow_df = pd.DataFrame(bow_data, index=self.tittles)
        
        # Add labels as a new column if provided
        bow_df['Label'] = self.labels
        
        return bow_df