import streamlit as st
import os
import streamlit as st
import tempfile
import tiktoken
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def compute_number_of_tokens(string: str, encoding_name: str = "gpt2") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in French:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50,length_function = compute_number_of_tokens)
chain = LLMChain(llm=llm, prompt=PROMPT, verbose=True)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_answer(question, docsearch):
    docs = docsearch.similarity_search(question, k=5)
    resp = chain.run({"context": docs, "question": question})
    return resp
st.title("chatbot NUKEMA 😊")
uploaded_file = st.file_uploader("Drag and drop un fichier PDF", type=["pdf"])
question = st.text_input("Posez votre question ici")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        tmpfile_path = tmpfile.name

    loader = UnstructuredPDFLoader(tmpfile_path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    docsearch = FAISS.from_documents(texts, embeddings)

# Make sure both the file is uploaded and the question is not empty before proceeding.
if question and uploaded_file is not None:  
    answer = get_answer(question, docsearch)
    st.write("Réponse :")
    st.write(answer)

