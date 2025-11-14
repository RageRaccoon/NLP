import pandas as pd
from math import log
from .Tokenizer import Tokenizer

class TF_IDF(Tokenizer):
    """ Class for creating the TF-IDF matrix """
    
    """ Constructor """
    def __init__(self):
        # Initialize the parent Tokenizer class
        super().__init__() 
        
        self.documents = []
        self.tittles = []
        self.labels = []
        self.tokens = []
        self.vocabulary = set()

    # Init all variables
    def fit_transform(self, docs:list, tittles:list=None, labels:list=None):
        self.documents = docs
        self.tittles = tittles if tittles is not None else [f"Document {i+1}" for i in range(len(docs))]
        self.labels = labels if labels is not None else [f"Doc_{i+1}" for i in range(len(docs))]
        self.tokens = []
        self.vocabulary = set()
        
        # Tokenize each document and build vocabulary
        for doc in self.documents:
            doc_tokens = self.tokenize(doc)
            self.tokens.append(doc_tokens)
            self.vocabulary.update(doc_tokens)

        # Convert vocabulary to sorted list for consistent column order
        self.vocabulary = sorted(list(self.vocabulary))
    
    """ Methods """
    # Compute term frequency for a given token list
    def compute_tf(self, token_list: list) -> pd.Series:
        # Create a Series with vocabulary as index, initialized to 0
        tf = pd.Series(0, index=self.vocabulary)
        
        # Count occurrences of each word
        for word in token_list:
            if word in tf.index:
                tf[word] += 1
        
        return tf
    
    # Compute inverse document frequency for the entire corpus
    def compute_idf(self) -> pd.Series:
        N = len(self.documents)
        idf = pd.Series(0.0, index=self.vocabulary)
        
        for word in self.vocabulary:
            # Count how many documents contain this word
            doc_count = sum(1 for doc_tokens in self.tokens if word in doc_tokens)
            # Calculate IDF using the smoothed formula: log(N / (1 + doc_count))
            idf[word] = log(N / (1 + doc_count))
        
        return idf

    # Compute the TF-IDF matrix
    def compute_tf_idf(self):
        # Compute TF for each document
        tf_matrix = []
        for i, doc_tokens in enumerate(self.tokens):
            tf_series = self.compute_tf(doc_tokens)
            tf_matrix.append(tf_series)
        
        # Create TF DataFrame
        tf_df = pd.DataFrame(tf_matrix, index=self.tittles)
        
        # Compute IDF
        idf_series = self.compute_idf()
        
        # Compute TF-IDF by multiplying TF matrix with IDF vector
        tf_idf_matrix = tf_df.multiply(idf_series, axis=1)
        
        # Add labels at the final column of the DataFrame
        tf_idf_matrix['Label'] = self.labels
        
        # Remove columns (words) that are all 0
        matrix_withoutZeroes = tf_idf_matrix.loc[:, (tf_idf_matrix != 0).any(axis=0)]
        
        return matrix_withoutZeroes