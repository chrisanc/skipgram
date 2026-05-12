import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Tokenizer:
    def __init__(self):
        self.file = self.__read_file_content(
            filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        )

    def __read_file_content(self, filepath: str) -> str:
        with open(filepath, "r") as f:
            content = f.read().upper()
        
        return self.__remove_punctuation(content)
    
    def __remove_punctuation(self, value: str) -> list[str]:
        return "".join(re.findall(r"[\w\s*]+", value))
    
    def get_tokens(self) -> np.ndarray:
        stop_words = stopwords.words("english")
        tokens = word_tokenize(self.file)
        tokens = [value.upper() for value in tokens if value not in stop_words]
        tokens = pd.unique(tokens)

        return tokens
    
    def one_hot_encoding(self, tokens: np.ndarray) -> pd.DataFrame:
        if tokens.size == 0:
            tokens = self.get_tokens()

        zeros = np.zeros((len(tokens), len(tokens)))
        for i in range(len(tokens)):
            zeros[i][i] = 1
        df = pd.DataFrame(data=zeros, columns=tokens)

        return df
    
    def create_pairs(self, one_hot: pd.DataFrame, tokens: np.ndarray, slidingWindow: int = 2) -> list[tuple[int, int]]:
        results: list[tuple[int, int]] = list()
        sumIds = 0
        for i in range(0, len(tokens)):
            subdf = one_hot.iloc[:, i-sumIds:i + slidingWindow + 1]
            for j in range(len(subdf.columns)):
                if tokens[i] == subdf.columns[j]:
                    continue
                results.append((i, i - sumIds + j))

            # Add one to the sumIds if necessary
            if sumIds < slidingWindow:
                sumIds+=1

        return results
    

    def tf_idf(self, tokens: np.ndarray = None):
        """
        Calculates the TF-IDF for each one of the tokens
        based on the corpus
        """
        if (tokens == None):
            tokens = self.get_tokens()
            
        docs = self.file.split("\n\n")
        freq = dict()
        tf_idf = pd.DataFrame(data=[], columns=tokens)
        
        # Fill the map with the amount of documents containing each token
        for token in tokens:
            total = 0
            # For get the occurences of each tokens in all documents
            for doc in docs:
                if doc.count(token) > 0:
                    total+=1
            
            freq.update({token: total})
        
        # Calculate the 
        for doc in docs:
            row = list()
            for token in tokens:
                tf = doc.count(token)
                idf = np.log(len(docs) / freq.get(token, 0.01))
                row.append(tf * idf)
            
            tf_idf.loc[len(tf_idf)] = row
            
        return tf_idf