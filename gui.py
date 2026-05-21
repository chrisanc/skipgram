import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from adapters.tokenizer import Tokenizer
from adapters.embeddings import SkipGram
from corpus import Corpus

langs = {
    "Inglés": "english",
    "Español": "spanish"
}

class GUI:
    def __init__(self, corpus: Corpus, skipgram: SkipGram):
        self.__corpus = corpus
        self.__skipgram = skipgram
        self.__remove_punctuation = lambda x: "".join(re.findall(r"[\w\s]+", x)).upper()
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
            file:str = self.__remove_punctuation(file.getvalue().decode("utf-8"))
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
            self.__corpus.update_corpus(file, langs[selected_lang], sliding_window)
            st.session_state.corpus = self.__corpus

            # Display all the vocabulary
            st.subheader("Vocabulario del corpus")
            st.write(f"Total de palabras: {len(self.__corpus.tokens)}")
            st.dataframe(self.__corpus.tokens)
            
            # Display all the pairs
            st.subheader(f"Pares creados con la ventana de contexto en {sliding_window}")
            st.write(f"Total de pares: {len(self.__corpus.pairs)}")
            st.dataframe(
                pd.DataFrame(
                    data=[[self.__corpus.tokens[i], self.__corpus.tokens[j]] for i, j in self.__corpus.pairs],
                    columns=["Palabra 1", "Palabra 2"]
                ),
                hide_index=True
            )
            
            # Obtain the TF-IDF index from the data
            tf_idf = self.__corpus.tokenizer.tf_idf(rows, self.__corpus.tokens)
            # Display the TF-IDF results
            self.__plot_tf_idf(tf_idf)
            
            if st.button("Entrenar SkipGram"):
                self.__skipgram.embeddings(self.__corpus.tokens, self.__corpus.one_hot, self.__corpus.pairs, 0)
        else:
            # Get the trained embeddings
            WIn, WOut = np.load("/home/chris/Documents/ProyectosGithub/skipgram/objects/WIn.npy"), np.load("/home/chris/Documents/ProyectosGithub/skipgram/objects/WOut.npy")
            # Reduct the dimensionality
            pca_model = PCA(n_components=3)
            embedding = pca_model.fit_transform(WIn)
            
            # Execute the skipgram algorithm. Each token has a embedding
            figure = go.Figure(
                data=[
                    go.Scatter3d(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=embedding[:, 2],
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        hovertext=st.session_state.corpus.tokens
                    )
                ]
            )
            figure.update_layout(
                title="Espacio vectorial semántico de los tokens",
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3"
                )
            )
            st.plotly_chart(figure)
            
            # Semantic search by an input
            st.subheader("Busqueda semantica por palabra")
            search_word = st.text_input("Ingresa una palabra", value="")
            # Verify it's just one word
            if search_word == "":
                return
            search_word = search_word.strip()
            if search_word.find(" ") > 0:
                st.warning("Debe ser una sola palabra!...")
                return
            
            # Execute the data search
            i = self.__skipgram.semantic_lookup(st.session_state.corpus.one_hot.iloc[8, :], WIn, WOut)
            print(st.session_state.corpus.tokens[8], st.session_state.corpus.tokens[i])
            st.write(f"Palabra de continuación más probable: {st.session_state.corpus.tokens[i]}")
            
            
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