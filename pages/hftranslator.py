import streamlit as st
import numpy as np
import torch
from transformers import pipeline
from transformers import AutoTokenizer

device = "cpu"

st.title("Huggingface Translator")

# setting up the dropdown list of the languages

option1 = st.selectbox(
    'Which language would you choose to type',
    ('English', 'Russian', 'German', 'French', 'Spanish', 'Japanese'))

option2 = st.selectbox('Which language would you like to translate to',
                       ('German','English','Russian','German', 'French', 'Spanish', 'Japanese'))

sent = "Enter the text in the "+option1 +" language below"

sentence = st.text_area(sent, height=250)

transformer = None

langmap = { 'English' : 'en',
            'Russian' :  'ru',
            'French'  :  'fr',
            'German'  : 'de',
            'Spanish' : 'es',
            'Japanese': 'jap'}

lang1 = langmap[option1]
lang2 = langmap[option2]

if option1 != option2:
    translator = pipeline("translation",
                      model=f'Helsinki-NLP/opus-mt-{lang1}-{lang2}', device=0)


if st.button("Translate"):
    if option1 == option2:
        st.write("Please Select different language for translation")
    else:
        ans = translator(sentence)[0]['translation_text']
        st.write(ans)
