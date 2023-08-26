import streamlit as st

import file_function as ffunc
import prompt_function as pfunc

LLM, EMBED_MODEL, PROMPT = pfunc.load_model()

def get_answer(question, docsearch):
    qa = pfunc.load_retrieval(llm = LLM, docsearch=docsearch, prompt=PROMPT)
    result = qa({"query": question})
    return result

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.image("Delos.png", width=150)

st.title("Delos-GPT")
uploaded_file = st.file_uploader("Déposez un fichier PDF", type=["pdf"])
question = st.text_input("Posez votre question ici")

if uploaded_file is not None:
    docsearch = pfunc.create_vector_database(uploaded_file, EMBED_MODEL)

if question and uploaded_file is not None:  
    result = get_answer(question, docsearch)
    print(result['source_documents'])
    print(str(len(result['source_documents'])))
    st.write(str(result['result']))

