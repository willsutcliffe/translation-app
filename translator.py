import streamlit as st
import numpy as np
import torch
from transformers import pipeline
from translation.models.encoder import Encoder
from translation.models.decoder import Decoder
from translation.models.transformers import Transformer
from transformers import AutoTokenizer

device = "cpu"
st.title("Self-trained Transformer Translator")

st.write("""I trained this deep learning translator with the original Transformer architecture from the paper 'Attention 
         is all you need'. Click on hftranslator to use a pre-trained Huggingface translation model with more languages.""")
def initialize_model(tokenizer):
    encoder = Encoder(vocab_size=tokenizer.vocab_size + 1,
                      max_len=60,
                      d_key=64,
                      d_model=512,
                      n_heads=8,
                      n_layers=4,
                      dropout_prob=0.1)
    decoder = Decoder(vocab_size=tokenizer.vocab_size + 1,
                      max_len=60,
                      d_key=64,
                      d_model=512,
                      n_heads=8,
                      n_layers=4,
                      dropout_prob=0.1)
    transformer = Transformer(encoder, decoder)
    return transformer


def translate(input_sentence,model, tokenizer):
    enc_input = tokenizer(input_sentence, return_tensors='pt').to(device)
    enc_output = model.encoder(enc_input['input_ids'], enc_input['attention_mask'])

    dec_input_ids = torch.tensor([[ int(tokenizer.vocab_size)]], device=device)
    dec_attn_mask = torch.ones_like(dec_input_ids, device=device)

    for _ in range(60):
        dec_output = model.decoder(
            enc_output,
            dec_input_ids,
            enc_input['attention_mask'],
            dec_attn_mask,
        )

        prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

        dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))

        dec_attn_mask = torch.ones_like(dec_input_ids)

        if prediction_id == 0:
            break
  
    translation = tokenizer.decode(dec_input_ids[0, 1:])
    #print(translation)
    return(translation)


# setting up the dropdown list of the languages

option1 = st.selectbox(
    'Which language would you choose to type',
    ('English', 'Russian', 'German', 'French'))

option2 = st.selectbox('Which language would you like to translate to',
                       ('English','Russian', 'German', 'French'))

langmap = { 'English' : 'en',
            'Russian' :  'ru',
            'French'  :  'fr',
            'German'  : 'de'
           }

lang1 = langmap[option1]
lang2 = langmap[option2]


sent = "Enter the text in the "+option1+" language below"

if option1 == option2:
    st.write("Choose a language pair please.")
else:
    model_checkpoint = f"Helsinki-NLP/opus-mt-{lang1}-{lang2}"
    torch_model = f"assets/{lang1}-{lang2}-final.pt"
    transformer = None
    # setting up the dictionary of languages to their keywords
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device = device)
    transformer = initialize_model(tokenizer)
    transformer.load_state_dict(torch.load(torch_model,  map_location="cpu"))
    transformer.eval()
    transformer.to(device)

sentence = st.text_area(sent, height=250)

if st.button("Translate"):
    if option1 == option2:
        st.write("Please Select different language for translation")
    elif transformer == None:
        st.write("Right now one language must be English.")
    else:
        ans = translate(sentence, transformer, tokenizer)[:-4]
        st.write(ans)
