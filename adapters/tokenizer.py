import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

class Tokenizer:
    def remove_punctuation(self, value: str) -> str:
        return "".join(re.findall(r"[\w\s]+", value))
    
    def get_tokens(self, file:str, language:str = "english") -> np.ndarray:
        stop_words = stopwords.words(language)
        tokens = np.array([value for value in file.split() if value.lower() not in stop_words and value != ""])
        tokens = pd.unique(tokens)

        return tokens
    
    def one_hot_encoding(self, tokens: np.ndarray) -> pd.DataFrame:
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
    

    def tf_idf(self, docs: list[str], tokens: np.ndarray = None) -> pd.DataFrame:
        """
        Calculates the TF-IDF for each one of the tokens
        based on the corpus
        """ 
        freq = dict()
        
        # Fill the map with the amount of documents containing each token
        for token in tokens:
            total = 0
            # For get the occurences of each tokens in all documents
            for doc in docs:
                if doc.count(token) > 0:
                    total+=1
            
            freq.update({token: total})
        
        # Calculate the TF-IDF index for each document
        scores = list()
        for doc in docs:
            row = list()
            for token in tokens:
                tf = doc.count(token)
                idf = np.log(len(docs) / freq.get(token, 0.01))
                row.append(tf * idf)
            
            scores.append(row)
            
        return pd.DataFrame(data=scores, columns=tokens)