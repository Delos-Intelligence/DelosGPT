import streamlit as st

import file_function as ffunc
import prompt_function as pfunc

LLM, EMBED_MODEL, PROMPT = pfunc.load_model()

def get_answer(question, docsearch):
    qa = pfunc.load_retrieval(llm = LLM, docsearch=docsearch, prompt=PROMPT)
    result = qa({"query": question})
    return result

st.title("Chatbot NUKEMA")
uploaded_file = st.file_uploader("Drag and drop un fichier PDF", type=["pdf"])
question = st.text_input("Posez votre question ici")

if uploaded_file is not None:
    docsearch = pfunc.create_vector_database(uploaded_file, EMBED_MODEL)

if question and uploaded_file is not None:  
    result = get_answer(question, docsearch)
    st.write(str(result['result']) + '. Pour plus d\'informations, consulter la page '+str(result['source_documents'][0].metadata['page']))

