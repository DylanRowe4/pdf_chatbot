import json
import os
import urllib
import requests

#import packages needed for openai llm
import openai
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

#import streamlit
import streamlit as st

#load our open ai information
with open('keys/hf.json', 'r') as rd_f:
    data = json.load(rd_f)
    
os.environ['HUGGINGFACEHUB_API_TOKEN'] = data['HUGGINGFACEHUB_API_TOKEN']

#load llm model and embeddings
llm = HuggingFaceHub(repo_id='google/flan-t5-large',
                     model_kwargs={"temperature":0.5, "max_length":512},
                     verbose=True)
embeddings = HuggingFaceEmbeddings()

#load pre-created document store
document_store = FAISS.load_local('LoTR', embeddings)

# expose this index in a retriever interface
retriever = document_store.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":10})
#create qa system with source document retrieval
qa_system = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

#start streamlit instance
st.title('Lord of The Rings Search Engine')
#create a text input box for the user
prompt = st.text_input('Input your question here.')

if prompt:
    #pass the prompt to the LLM
    response = qa_system({"query": prompt})
    #write it out to the screen
    st.write(f"Answer:\n{response['result']}\n" + '*'*50)
    for page in response['source_documents']:
        st.write(f"Page {page.metadata['page']}:\n\n{page.page_content}")
        st.write('-'*100)
