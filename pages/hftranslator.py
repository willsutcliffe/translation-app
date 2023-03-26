import streamlit as st
import numpy as np
import torch
from transformers import pipeline
from transformers import AutoTokenizer

device = "cpu"

st.title("Huggingface Translator")

# setting up the dropdown list of the languages

st.write("""This translator use pre-trained This translator use pre-trained models from Huggingface Helsinki-NLP/opus-mt-X-Y.
         The models must first be downloaded hence the delay.""")
option1 = st.selectbox(
    'Which language would you choose to type',
    ('English', 'Russian', 'German', 'French', 'Spanish', 'Japanese', 'Chinese'))

option2 = st.selectbox('Which language would you like to translate to',
                       ('German','English','Russian','German', 'French', 'Spanish', 'Japanese', 'Chinese'))

sent = "Enter the text in the "+option1 +" language below"

sentence = st.text_area(sent, height=250)

transformer = None

langmap = { 'English' : 'en',
            'Russian' :  'ru',
            'French'  :  'fr',
            'German'  : 'de',
            'Spanish' : 'es',
            'Japanese': 'jap',
            'Chinese' : 'zh'}

lang1 = langmap[option1]
lang2 = langmap[option2]

if option1 != option2:
    translator = pipeline("translation",
                      model=f'Helsinki-NLP/opus-mt-{lang1}-{lang2}', device='cpu')


if st.button("Translate"):
    if option1 == option2:
        st.write("Please Select different language for translation")
    else:
        ans = translator(sentence)[0]['translation_text']
        st.write(ans)
