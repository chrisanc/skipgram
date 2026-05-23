import pandas as pd
import numpy as np
from adapters.tokenizer import Tokenizer
import random

class SkipGram:
    def __init_weights(self, rows: int, cols: int) -> np.ndarray:
        # Random values between 0 and 1 inclusive
        return np.random.random((rows, cols))
    
    # Math functions definition
    def __softmax(self, Z: np.ndarray):
        return np.exp(Z) / np.sum(np.exp(Z))
    
    
    def __cosine_similarity(self, embedding: pd.Series, other: pd.Series) -> float:
        return (embedding @ other) / (self.__norm_magnitude(embedding) * self.__norm_magnitude(other))
    
    
    def __norm_magnitude(self, embedding: pd.Series) -> float:
        return np.sqrt(np.sum(embedding ** 2))
    
    
    def embeddings(
        self, tokens: np.ndarray, one_hot: pd.DataFrame, pairs: list[tuple[int, int]],
        word_index: int, embedding_size: int = 300, learning_rate: float = 0.01, epochs:int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the weights matrix (embeddings) to work with based on a small
        Neural Network (SkipGram).
        """
        try:
            return (
                np.load("/home/chris/Documents/github-projects/skipgram/objects/WIn.npy"),
                np.load("/home/chris/Documents/github-projects/skipgram/objects/WOut.npy")
            )
        except OSError:
            print("The file doesn't exists, generating...")
        
        # Weights initialization
        input_weights = self.__init_weights(len(tokens), embedding_size)
        output_weights = self.__init_weights(embedding_size, len(tokens))
        
        target_words, context_words = zip(*pairs)
        
        # Start the training loop by epochs
        for epoch in range(epochs + 1):
            total_loss = 0
            # On each epoch, we must apply a whole process...
            for target_word, context_word in zip(target_words, context_words):
                # Apply the forward pass
                input_vector = input_weights[target_word]       
                output_vector = np.dot(input_vector, output_weights)
                output_probs = self.__softmax(output_vector)
                
                # Calculate the gradient
                error = output_probs.copy()
                error[context_word] -= 1
                
                # Backpropagation: Weights adjustment
                input_grad = np.dot(output_weights, error)
                output_grad = np.outer(input_vector, error)
                # Update the weights
                input_weights[target_word] -= learning_rate * input_grad
                output_weights -= learning_rate * output_grad
                
                # Calculate the loss on this iteration
                total_loss += -np.log(output_probs[context_word])
            
            if epoch % (epochs / 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs)}")

        # Save the matrices in binary NumPy files
        np.save(file="/home/chris/Documents/github-projects/skipgram/objects/WIn.npy", arr=input_weights)
        np.save(file="/home/chris/Documents/github-projects/skipgram/objects/WOut.npy", arr=output_weights)
        # Return the matrices
        return input_weights, output_weights
    

    def semantic_lookup(self, token_index: int, context_window: int, input_weights: np.ndarray, tokens: np.ndarray):
        """
        Performs the semantic search based on the trained weights
        """
        # Obtain the embedding of the word
        word_embedding = input_weights[token_index]
        # Create a new array with the data
        data = list()
        # Obtain the similarity per row
        for i, row in enumerate(input_weights):
            if i == token_index:
                continue
            
            data.append([tokens[i], self.__cosine_similarity(word_embedding, row)])
        # Create a dataframe with the given data ordered
        similarity = pd.DataFrame(columns=["Token", "Similarity"], data=data, index=None)
        # Order in descending order
        similarity.sort_values(by="Similarity", ascending=False, inplace=True)
        
        # Based on the context window, we retrieve the N closest elements
        return similarity[max(0, token_index - context_window):min(len(tokens) - 1, token_index + context_window)]