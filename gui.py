import streamlit as st
import pandas as pd
import numpy as np
from adapters.tokenizer import Tokenizer
from adapters.embeddings import SkipGram

langs = {
    "Inglés": "english",
    "Español": "spanish"
}

class GUI:
    def __init__(self, tokenizer: Tokenizer, skipgram: SkipGram):
        self.tokenizer = tokenizer
        self.skipgram = skipgram
        self.__setup()
    
    def __setup(self):
        st.title("Minería de Texto y Busqueda Semantica")
        st.subheader("TF-IDF y Skipgram")
        file = st.file_uploader("Ingresa el corpus", accept_multiple_files=False, type=["txt"])
        if file == None:
            return
    
        # Create the sidebar contents
        # Add to a select box the two main sections
        st.sidebar.subheader("Zona de filtros")
        section = st.sidebar.selectbox("Selecciona la sección...", options=["Minería de Texto", "SkipGram"])
    
        if section == "Minería de Texto":
            # Extract and normalize the file content
            file:str = self.tokenizer.remove_punctuation(file.getvalue().decode("utf-8")).upper()
            # Set up sidebar content
            selected_lang = st.sidebar.selectbox("Selecciona el idioma de tu corpus", options=langs.keys())
            sliding_window = st.sidebar.slider(
                "Selecciona el tamaño de la ventana de contexto", min_value=1, max_value=20, value=2
            )
            split_method = st.sidebar.selectbox(
                "Como identificarás los documentos en tu corpus?",
                options=["Por parrafos", "Por filas"]
            )
            if split_method == "Por filas":
                # Calculate the amount of rows
                raw_rows = [value.replace("\r", "").strip() for value in file.split("\n") if value.replace("\r", "").strip() != ""]
                amount_rows = st.sidebar.slider(
                    "Selecciona la cantidad de filas por documento",
                    min_value=5, max_value=len(raw_rows),
                    value=40
                )

                # Create the fixed-size rows
                rows = ["\n".join(raw_rows[index-amount_rows:index]) for index in range(amount_rows, len(raw_rows) + amount_rows, amount_rows)]
            else:
                rows = [value for value in file.splitlines() if value != ""]
            
            # Execute the tokenizer methods
            tokens = self.tokenizer.get_tokens(file, language=langs[selected_lang])
            one_hot = self.tokenizer.one_hot_encoding(tokens)
            pairs = self.tokenizer.create_pairs(one_hot, tokens, slidingWindow=sliding_window)

            # Display all the vocabulary
            st.subheader("Vocabulario del corpus")
            st.write(f"Total de palabras: {len(tokens)}")
            st.dataframe(tokens)
            
            # Display all the pairs
            st.subheader(f"Pares creados con la ventana de contexto en {sliding_window}")
            st.write(f"Total de pares: {len(pairs)}")
            st.dataframe(
                pd.DataFrame(
                    data=[[tokens[i], tokens[j]] for i, j in pairs],
                    columns=["Palabra 1", "Palabra 2"]
                ),
                hide_index=True
            )
            
            # Obtain the TF-IDF index from the data
            tf_idf = self.tokenizer.tf_idf(rows, tokens)
            # Display the TF-IDF results
            self.__plot_tf_idf(tf_idf)
        else:
            st.title("Skipgram y busqueda semantica")
            # Execute the skipgram algorithm. Each token has a embedding
            # embeds = self.skipgram.embeddings(tokens, one_hot, pairs, 0)
            
            
    def __plot_tf_idf(self, tf_idf: pd.DataFrame):
        df = pd.DataFrame(columns=["DOCUMENTO", "PALABRAS RELEVANTES"])
        st.subheader("RESULTADOS DEL TF-IDF")
        st.write(f"Total de documentos: {len(tf_idf)}")
        # Display the data as a dataframe
        df = tf_idf.apply(self.__extract_values, axis=1)
        st.dataframe(data=df, hide_index=True)
        
        
    def __extract_values(self, row: pd.Series) -> pd.Series:
        """
        Extracts the values for each row for display purposes
        """
        values = row.sort_values(ascending=False).head(3)
        
        return pd.Series({
            "DOCUMENTO": row.name + 1,
            "PALABRAS RELEVANTES": " ".join([f"{name} ({score:.2f})" for name, score in values.items()])
        })