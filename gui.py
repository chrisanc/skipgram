import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from adapters.tokenizer import Tokenizer
from adapters.embeddings import SkipGram

langs = {
    "Español": "spanish",
    "Inglés": "english"
}

class GUI:
    def __init__(self, tokenizer: Tokenizer, skipgram: SkipGram):
        self.__tokenizer = tokenizer
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
        
        # Extract and normalize the file content
        file:str = self.__remove_punctuation(file.getvalue().decode("utf-8"))
        # Set up sidebar content
        selected_lang = st.sidebar.selectbox("Selecciona el idioma de tu corpus", options=langs.keys())
        context_window = st.sidebar.slider(
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
        tokens = self.__tokenizer.get_tokens(file, langs[selected_lang])
        one_hot = self.__tokenizer.one_hot_encoding(tokens)
        pairs = self.__tokenizer.create_pairs(tokens, context_window)

        st.title("Minería de Texto")

        # Display all the vocabulary
        st.subheader("Vocabulario del corpus")
        st.write(f"Total de palabras: {len(tokens)}")
        st.dataframe(tokens)

        # Display the one hot
        st.subheader("One-hot encoding de los datos")
        # Set the tokens as the index
        one_hot.set_index(tokens, inplace=True)
        st.dataframe(one_hot)
        
        # Display all the pairs
        st.subheader(f"Pares creados con la ventana de contexto en {context_window}")
        st.write(f"Total de pares: {len(pairs)}")
        st.dataframe(
            pd.DataFrame(
                data=[[tokens[i], tokens[j]] for i, j in pairs],
                columns=["Palabra 1", "Palabra 2"]
            ),
            hide_index=True
        )
        
        # Obtain the TF-IDF index from the data
        tf_idf = self.__tokenizer.tf_idf(rows, tokens)
        # Display the TF-IDF results
        self.__plot_tf_idf(tf_idf)
            
        st.title("Skipgram")
        # Get the trained embeddings
        try:
            WIn, WOut = np.load("/home/chris/Documents/github-projects/skipgram/objects/WIn.npy"), np.load("/home/chris/Documents/github-projects/skipgram/objects/WOut.npy")
        except Exception:
            st.warning("No se encuentran los pesos de entrenamiento...")
            if st.button("Entrenar SkipGram"):
                self.__skipgram.embeddings(tokens, one_hot, pairs, 0)
            return
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
                    hovertext=tokens
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
        
        # Execute the data search
        st.subheader("Busqueda semantica por token de corpus")
        selected_token = st.selectbox("Selecciona un token", options=tokens)
        token_index = np.where(tokens == selected_token)[0][0]
        cosine_similarity = self.__skipgram.semantic_lookup(
            token_index, context_window, WIn, tokens
        )
        
        st.dataframe(cosine_similarity, hide_index=True)
            
    def __plot_tf_idf(self, tf_idf: pd.DataFrame):
        st.subheader("RESULTADOS DEL TF-IDF")
        st.write(f"Total de documentos: {len(tf_idf)}")
        # Display the data as a dataframe
        st.dataframe(data=tf_idf, hide_index=False)