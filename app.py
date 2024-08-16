import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Load the model and tokenizers
model = load_model('text_summarization_gru_best_model_final.keras')
with open('tokenizer_article.pkl', 'rb') as f:
    tokenizer_article = pickle.load(f)
with open('tokenizer_summary.pkl', 'rb') as f:  # Remove the extra parenthesis here
    tokenizer_summary = pickle.load(f)

# Constants
max_text_len = 100


# Function to summarize text
def summarize_text(article):
    sequence = tokenizer_article.texts_to_sequences([article])
    padded_sequence = pad_sequences(sequence, maxlen=max_text_len, padding='post')
    prediction = model.predict(padded_sequence)
    summary_sequence = np.argmax(prediction[0], axis=1)
    summary = ' '.join([tokenizer_summary.index_word[i] for i in summary_sequence if i != 0])
    return summary

# Streamlit UI
st.title('Text Summarization App')
article = st.text_area('Enter the article:')
if st.button('Summarize'):
    summary = summarize_text(article)
    st.write('Summary:', summary)
