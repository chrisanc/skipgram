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
        return np.random.random(rows, cols)
    
    # Math functions definition
    def __softmax(Z: np.ndarray):
        return np.exp(Z) / np.sum(np.exp(Z))
    
    def __cross_entropy_loss(A1: np.ndarray) -> float:
        pass
    
    def embeddings(self, epochs:int = 10000) -> np.ndarray:
        """
        Returns the weights matrix (embeddings) to work with based on a small
        Neural Network (SkipGram)
        """
        tokens = self.tokenizer.get_tokens()
        one_hot = self.tokenizer.one_hot_encoding(tokens)
        pairs = self.tokenizer.create_pairs(one_hot, tokens)

        for epoch in range(epochs + 1):
            pass

        return None