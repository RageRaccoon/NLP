"""NLP Co-occurrence Matrix with PCA Implementation"""

### This module provides a CoOccurrencePCA class for creating a co-occurrence matrix 
### and applying Principal Component Analysis (PCA) for dimensionality reduction.

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from .Tokenizer import Tokenizer

class CoOccurrencePCA(Tokenizer):
    """ Class for creating co-occurrence matrix and applying PCA """
    
    """ Constructor """
    def __init__(self, window_size:int=2):
        super().__init__()
        self.documents = []
        self.tittles = []
        self.labels = []
        self.tokens = []
        self.vocabulary = []
        self.window_size = window_size
        self.cooccurrence_matrix = None
        self.pca_model = None
        self.reduced_matrix = None
        
    
    def fit_transform(self, docs:list, tittles:list=None, labels:list=None, window_size:int=None):
        """
        Initialize the model with documents and parameters
        
        Parameters:
        -----------
        docs : list
            List of text documents to process
        tittles : list, optional
            List of titles for each document
        labels : list, optional
            List of labels for each document
        window_size : int, optional
            Size of the context window for co-occurrence (default: 2)
        """
        self.documents = docs
        self.tittles = tittles if tittles is not None else [f"Document {i+1}" for i in range(len(docs))]
        self.labels = labels if labels is not None else [f"Doc_{i+1}" for i in range(len(docs))]
        
        if window_size is not None:
            self.window_size = window_size
        
        self.tokens = []
        self.vocabulary = set()
        
        # Tokenize each document and build vocabulary
        for doc in self.documents:
            doc_tokens = self.tokenize(doc)
            self.tokens.append(doc_tokens)
            self.vocabulary.update(doc_tokens)
        
        # Convert vocabulary to sorted list
        self.vocabulary = sorted(list(self.vocabulary))
    
    
    """ Methods """
    def compute_cooccurrence_matrix(self) -> pd.DataFrame:
        """
        Creates a co-occurrence matrix based on word proximity within a window
        
        Returns:
        --------
        pd.DataFrame
            Co-occurrence matrix with words as both rows and columns
        """
        vocab_size = len(self.vocabulary)
        word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        # Initialize co-occurrence matrix
        cooccurrence = np.zeros((vocab_size, vocab_size), dtype=int)
        
        # Process each document's tokens
        for doc_tokens in self.tokens:
            # Slide window over tokens
            for i, target_word in enumerate(doc_tokens):
                if target_word not in word_to_idx:
                    continue
                    
                target_idx = word_to_idx[target_word]
                
                # Look at context words within window
                start = max(0, i - self.window_size)
                end = min(len(doc_tokens), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Don't count the word with itself
                        context_word = doc_tokens[j]
                        if context_word in word_to_idx:
                            context_idx = word_to_idx[context_word]
                            cooccurrence[target_idx, context_idx] += 1
        
        # Create DataFrame
        self.cooccurrence_matrix = pd.DataFrame(
            cooccurrence,
            index=self.vocabulary,
            columns=self.vocabulary
        )
        
        return self.cooccurrence_matrix
    
    
    def apply_pca(self, n_components:int=3) -> pd.DataFrame:
        """
        Applies PCA to reduce dimensionality of word vectors (rows) from the co-occurrence matrix
        Each word vector is reduced from vocabulary_size dimensions to n_components dimensions
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to keep (default: 3)
        
        Returns:
        --------
        pd.DataFrame
            Reduced matrix after PCA transformation with shape (vocabulary_size, n_components)
        """
        if self.cooccurrence_matrix is None:
            self.compute_cooccurrence_matrix()
        
        # Apply PCA to reduce each word vector from vocabulary_size to n_components
        self.pca_model = PCA(n_components=n_components)
        reduced_data = self.pca_model.fit_transform(self.cooccurrence_matrix.values)
        
        # Create DataFrame with reduced dimensions
        column_names = [f'PC{i+1}' for i in range(n_components)]
        self.reduced_matrix = pd.DataFrame(
            reduced_data,
            index=self.vocabulary,
            columns=column_names
        )
        
        return self.reduced_matrix
    
    
    def get_explained_variance(self) -> np.ndarray:
        """
        Returns the explained variance ratio of each principal component
        
        Returns:
        --------
        np.ndarray
            Array with variance explained by each component
        """
        if self.pca_model is None:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        
        return self.pca_model.explained_variance_ratio_
    
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Returns the cumulative explained variance
        
        Returns:
        --------
        np.ndarray
            Cumulative sum of explained variance
        """
        return np.cumsum(self.get_explained_variance())
    
    
    def get_word_vectors(self, words:list=None) -> pd.DataFrame:
        """
        Get PCA-reduced vectors for specific words
        
        Parameters:
        -----------
        words : list, optional
            List of words to get vectors for. If None, returns all words.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with word vectors
        """
        if self.reduced_matrix is None:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        
        if words is None:
            return self.reduced_matrix
        
        # Filter for requested words that exist in vocabulary
        available_words = [w for w in words if w in self.vocabulary]
        return self.reduced_matrix.loc[available_words]
    
    
    def get_similar_words(self, target_word:str, top_n:int=5) -> pd.DataFrame:
        """
        Find most similar words based on cosine similarity in PCA space
        
        Parameters:
        -----------
        target_word : str
            Word to find similar words for
        top_n : int, optional
            Number of similar words to return (default: 5)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with similar words and their similarity scores
        """
        if self.reduced_matrix is None:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        
        if target_word not in self.vocabulary:
            raise ValueError(f"Word '{target_word}' not in vocabulary")
        
        # Get target word vector
        target_vector = self.reduced_matrix.loc[target_word].values
        
        # Calculate cosine similarity with all words
        similarities = {}
        for word in self.vocabulary:
            if word != target_word:
                word_vector = self.reduced_matrix.loc[word].values
                # Cosine similarity
                cos_sim = np.dot(target_vector, word_vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(word_vector) + 1e-10
                )
                similarities[word] = cos_sim
        
        # Sort by similarity and get top N
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return pd.DataFrame(sorted_words, columns=['Word', 'Similarity'])
    
    def get_most_relevant_words(self, top_n:int=20, method:str='frequency') -> list:
        """
        Get the most relevant words based on different criteria
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top words to return (default: 20)
        method : str, optional
            Method to determine relevance:
            - 'frequency': Words with highest total co-occurrence count
            - 'variance': Words with highest variance in PCA space
            - 'central': Words closest to the center in PCA space
        
        Returns:
        --------
        list
            List of most relevant words
        """
        if self.cooccurrence_matrix is None:
            self.compute_cooccurrence_matrix()
        
        if method == 'frequency':
            # Sum of co-occurrences for each word
            word_scores = self.cooccurrence_matrix.sum(axis=1)
            top_words = word_scores.nlargest(top_n).index.tolist()
            
        elif method == 'variance':
            if self.reduced_matrix is None:
                self.apply_pca(n_components=3)
            # Words with highest variance across PCA components
            word_variance = self.reduced_matrix.var(axis=1)
            top_words = word_variance.nlargest(top_n).index.tolist()
            
        elif method == 'central':
            if self.reduced_matrix is None:
                self.apply_pca(n_components=3)
            # Words closest to the centroid
            centroid = self.reduced_matrix.mean(axis=0)
            distances = np.linalg.norm(self.reduced_matrix.values - centroid.values, axis=1)
            word_distances = pd.Series(distances, index=self.vocabulary)
            top_words = word_distances.nsmallest(top_n).index.tolist()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'frequency', 'variance', or 'central'")
        
        return top_words
    
    
    def plot_3d(self, top_n:int=20, method:str='frequency', figsize:tuple=(12, 10), 
                annotate:bool=True, title:str=None):
        """
        Creates a 3D visualization of the most relevant words in PCA space
        
        Parameters:
        -----------
        top_n : int, optional
            Number of most relevant words to plot (default: 20)
        method : str, optional
            Method to select relevant words: 'frequency', 'variance', or 'central' (default: 'frequency')
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (12, 10))
        annotate : bool, optional
            Whether to annotate points with word labels (default: True)
        title : str, optional
            Custom title for the plot. If None, a default title is used
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the 3D plot
        """
        # Ensure PCA has been applied with 3 components
        if self.reduced_matrix is None or self.reduced_matrix.shape[1] < 3:
            self.apply_pca(n_components=3)
        
        # Get most relevant words
        relevant_words = self.get_most_relevant_words(top_n=top_n, method=method)
        
        # Filter data for relevant words
        plot_data = self.reduced_matrix.loc[relevant_words]
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x = plot_data['PC1'].values
        y = plot_data['PC2'].values
        z = plot_data['PC3'].values
        
        # Plot points
        scatter = ax.scatter(x, y, z, c=range(len(relevant_words)), 
                           cmap='viridis', s=100, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # Add labels for each point
        if annotate:
            for i, word in enumerate(relevant_words):
                ax.text(x[i], y[i], z[i], word, fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Set labels and title
        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_zlabel('PC3', fontsize=12, fontweight='bold')
        
        if title is None:
            variance = self.get_explained_variance()
            title = f'3D PCA Visualization of Top {top_n} Words (Method: {method})\n' \
                   f'Explained Variance: PC1={variance[0]:.2%}, PC2={variance[1]:.2%}, PC3={variance[2]:.2%}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Word Index', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    
    def plot_3d_interactive(self, top_n:int=20, method:str='frequency', title:str=None, show_vectors:bool=True):
        """
        Creates an interactive 3D visualization using Plotly of the most relevant words in PCA space
        Words are shown as vectors from the origin (0,0,0)
        
        Parameters:
        -----------
        top_n : int, optional
            Number of most relevant words to plot (default: 20)
        method : str, optional
            Method to select relevant words: 'frequency', 'variance', or 'central' (default: 'frequency')
        title : str, optional
            Custom title for the plot. If None, a default title is used
        show_vectors : bool, optional
            Whether to show vectors from origin to points (default: True)
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure object
        """
        # Ensure PCA has been applied with 3 components
        if self.reduced_matrix is None or self.reduced_matrix.shape[1] < 3:
            self.apply_pca(n_components=3)
        
        # Get most relevant words
        relevant_words = self.get_most_relevant_words(top_n=top_n, method=method)
        
        # Filter data for relevant words
        plot_data = self.reduced_matrix.loc[relevant_words]
        
        # Extract coordinates
        x = plot_data['PC1'].values
        y = plot_data['PC2'].values
        z = plot_data['PC3'].values
        
        # Create figure with traces
        traces = []
        
        # Add vectors from origin to each point
        if show_vectors:
            for i, word in enumerate(relevant_words):
                # Calculate color components
                r = int(255 * i / len(relevant_words))
                g = int(100 + 155 * (1 - i / len(relevant_words)))
                b = int(200 - 100 * i / len(relevant_words))
                
                # Vector line from origin to point
                traces.append(go.Scatter3d(
                    x=[0, x[i]],
                    y=[0, y[i]],
                    z=[0, z[i]],
                    mode='lines',
                    line=dict(color=f'rgb({r}, {g}, {b})', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add scatter points at the end of vectors
        traces.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=list(range(len(relevant_words))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Word Index", x=1.02),
                line=dict(color='white', width=1)
            ),
            text=relevant_words,
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hovertemplate='<b>%{text}</b><br>' +
                          'PC1: %{x:.3f}<br>' +
                          'PC2: %{y:.3f}<br>' +
                          'PC3: %{z:.3f}<br>' +
                          '<extra></extra>',
            name='Words'
        ))
        
        # Add origin point
        traces.append(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='Origin (0,0,0)',
            hovertemplate='<b>Origin</b><br>PC1: 0<br>PC2: 0<br>PC3: 0<extra></extra>'
        ))
        
        fig = go.Figure(data=traces)
        
        # Set layout and title
        if title is None:
            variance = self.get_explained_variance()
            title = f'3D PCA Word Vectors from Origin - Top {top_n} Words (Method: {method})<br>' \
                   f'Explained Variance: PC1={variance[0]:.2%}, PC2={variance[1]:.2%}, PC3={variance[2]:.2%}'
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis=dict(title='PC1', backgroundcolor="rgb(240, 240, 245)", 
                          gridcolor="white", showbackground=True, zeroline=True, zerolinewidth=2, zerolinecolor="red"),
                yaxis=dict(title='PC2', backgroundcolor="rgb(240, 240, 245)", 
                          gridcolor="white", showbackground=True, zeroline=True, zerolinewidth=2, zerolinecolor="red"),
                zaxis=dict(title='PC3', backgroundcolor="rgb(240, 240, 245)", 
                          gridcolor="white", showbackground=True, zeroline=True, zerolinewidth=2, zerolinecolor="red"),
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=60),
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    
    def visualize_summary(self) -> dict:
        """
        Returns a summary of the analysis
        
        Returns:
        --------
        dict
            Dictionary with analysis statistics
        """
        summary = {
            'num_documents': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'window_size': self.window_size,
            'cooccurrence_matrix_shape': self.cooccurrence_matrix.shape if self.cooccurrence_matrix is not None else None,
            'reduced_matrix_shape': self.reduced_matrix.shape if self.reduced_matrix is not None else None,
            'n_components': self.reduced_matrix.shape[1] if self.reduced_matrix is not None else None,
            'explained_variance': self.get_explained_variance().tolist() if self.pca_model is not None else None,
            'cumulative_variance': self.get_cumulative_variance().tolist() if self.pca_model is not None else None
        }
        
        return summary
        return summary
