import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = tf.keras.models.load_model('text_exploration.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max lengths based on your previous preprocessing
max_len_X = 500  # for articles
max_len_y = 50   # for highlights

# Define a function to generate the summary
def generate_summary(article):
    # Preprocess the article
    sequence = tokenizer.texts_to_sequences([article])
    padded_sequence = pad_sequences(sequence, maxlen=max_len_X, padding='post', truncating='post')

    # Generate summary
    decoder_input = tf.expand_dims([tokenizer.word_index['<OOV>']], 0)
    summary = []

    for _ in range(max_len_y):
        preds = model.predict([padded_sequence, decoder_input], verbose=0)
        predicted_id = tf.argmax(preds[0, -1, :]).numpy()
        
        if predicted_id == 0:  # Break if the padding token is predicted
            break

        predicted_word = tokenizer.index_word.get(predicted_id, '')
        summary.append(predicted_word)
        
        # Update decoder input with the predicted word
        decoder_input = tf.concat([decoder_input, [[predicted_id]]], axis=-1)

    return ' '.join(summary)

# Streamlit app layout
st.title("Text Summarization App")

# User input
user_input = st.text_area("Enter the article you want to summarize:")

if st.button("Generate Summary"):
    if user_input:
        summary = generate_summary(user_input)
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.error("Please enter an article to summarize.")