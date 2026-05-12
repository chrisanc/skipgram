import pandas as pd
import numpy as np
from tokenizer import Tokenizer
import random

class SkipGram:
    # Initialization functions
    def __init__(self):
        self.tokenizer = Tokenizer()

    def __init_weights(self, rows: int, cols: int) -> np.ndarray:
        # Random values between 0 and 1 inclusive
        return np.random.random((rows, cols))
    
    # Math functions definition
    def __softmax(self, Z: np.ndarray):
        return np.exp(Z) / np.sum(np.exp(Z))
    
    def __cross_entropy_loss(self, A1: np.ndarray, y: np.ndarray) -> float:
        return 0.0
    
    def embeddings(self, epochs:int = 10000) -> np.ndarray:
        """
        Returns the weights matrix (embeddings) to work with based on a small
        Neural Network (SkipGram).
        """
        tokens = self.tokenizer.get_tokens()
        one_hot = self.tokenizer.one_hot_encoding(tokens)
        pairs = self.tokenizer.create_pairs(one_hot, tokens)
        
        # Weights initialization
        W1 = self.__init_weights(len(tokens), 300)
        W2 = self.__init_weights(300, len(tokens))
        # Split the dataset in X and y, where X is the data and y the true labels
        X = one_hot.iloc[0, :]

        for epoch in range(epochs + 1):
            # Execute the forward pass
            Z1 = X @ W1
            Z2 = Z1 @ W2
            A2 = self.__softmax(Z2)
            
            # Calculate loss
            loss = self.__cross_entropy_loss(A2)
            
            # Backpropagation to get the results
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} with loss {loss:.2f}")
            
            # Backpropagation for learning
            break

        return W1