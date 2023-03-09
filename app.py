import streamlit as st
import numpy as np
import torch
from transformers import pipeline
from translation.models.encoder import Encoder
from translation.models.decoder import Decoder
from translation.models.transformers import Transformer
from transformers import AutoTokenizer

device = "cpu"

model_checkpoint = f"Helsinki-NLP/opus-mt-en-ru"
tokenizer_en_ru = AutoTokenizer.from_pretrained(model_checkpoint, device = device)

model_checkpoint = f"Helsinki-NLP/opus-mt-ru-en"
tokenizer_ru_en = AutoTokenizer.from_pretrained(model_checkpoint, device = device)

encoder = Encoder(vocab_size=tokenizer_en_ru.vocab_size + 1,
                  max_len=60,
                  d_key=64,
                  d_model=512,
                  n_heads=8,
                  n_layers=4,
                  dropout_prob=0.1)
decoder = Decoder(vocab_size=tokenizer_en_ru.vocab_size + 1,
                  max_len=60,
                  d_key=64,
                  d_model=512,
                  n_heads=8,
                  n_layers=4,
                  dropout_prob=0.1)
transformer_en_ru = Transformer(encoder, decoder)
transformer_en_ru.load_state_dict(torch.load('en-ru-final.pt',  map_location="cpu"))
transformer_en_ru.eval()
transformer_en_ru.to(device)


def translate(input_sentence,model, tokenizer):
  # get encoder output first
    enc_input = tokenizer(input_sentence, return_tensors='pt').to(device)
    #print(device)
    enc_output = model.encoder(enc_input['input_ids'], enc_input['attention_mask'])

    # setup initial decoder input
    dec_input_ids = torch.tensor([[ int(tokenizer.vocab_size)]], device=device)
    dec_attn_mask = torch.ones_like(dec_input_ids, device=device)

  # now do the decoder loop
    for _ in range(32):
        dec_output = model.decoder(
            enc_output,
            dec_input_ids,
            enc_input['attention_mask'],
            dec_attn_mask,
        )

        # choose the best value (or sample)
        prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

        # append to decoder input
        dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))

        # recreate mask
        dec_attn_mask = torch.ones_like(dec_input_ids)

        # exit when reach </s>
        if prediction_id == 0:
            break
  
    translation = tokenizer.decode(dec_input_ids[0, 1:])
    #print(translation)
    return(translation)

st.title("Language-Translator")

# setting up the dropdown list of the languages

option = st.selectbox(
    'Which language would you choose to type',
    ('English', 'Russian'))

option1 = st.selectbox('Which language would you like to translate to',
                       ('English','Russian'))


sent = "Enter the text in "+option+" language below"

# setting up the dictionary of languages to their keywords
if option == 'English' and option1 == 'Russian':
    model = transformer_en_ru
    tokenizer = tokenizer_en_ru
else:
    st.write("Unfortunately model not available yet. ")
    #model = transformer_ru_en
    #tokenizer = tokenizer_ru_en



sentence = st.text_area(sent, height=250)

if st.button("Translate"):
    if option == option1:
        st.write("Please Select different Language for Translation")

    else:
        
        ans = translate(sentence, model, tokenizer)[:-4]
        st.write(ans)
