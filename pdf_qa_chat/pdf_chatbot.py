import json
import os
import shutil
import tempfile
#import packages needed for openai llm
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
#import streamlit
import streamlit as st

def create_document_store(upload_files):
    #reset file path if multiple pdfs added
    temp_file_path = os.getcwd()
    
    
    #create temporary file path for pdf
    temp_dir = tempfile.TemporaryDirectory()
    for fil in upload_files:
        temp_file_path = os.path.join(temp_dir.name, fil.name)
        #write and save pdf file
        with open(temp_file_path, 'wb') as tmp:
            tmp.write(fil.read())
        
    #add pdf to loader object
    loader = PyPDFDirectoryLoader(temp_dir.name)
    #read pages of loader
    pages = loader.load() #use page[i].page_content to access page text if needed

    #set chunk and overlap size
    chunk_size = 850
    chunk_overlap = 150
    #create a text splitter using double new lines, then new lines, then periods, and so on if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   separators=['\n\n', '\n', '(?<=\. )', ' ', ''],
                                                   length_function=len)

    #split pages into chunks using our text splitter
    chunks = text_splitter.split_documents(pages)
    
    #get huggingface embeddings
    embeddings = HuggingFaceEmbeddings()

    #create chroma document store
    document_store = Chroma.from_documents(documents=chunks,
                                           embedding=embeddings) #persist_directory lets us save locally
    return document_store

def create_chain(document_store):
    #load llm model and embeddings
    llm = HuggingFaceHub(repo_id='google/flan-t5-large',
                     model_kwargs={"temperature":0.1, "max_length":512})
    
    #create memory object
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    #create conversation system with source document retrieval and our prompt
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     retriever=document_store.as_retriever(search_type="mmr",
                                                                                           search_kwargs={"k":5,
                                                                                                          "fetch_k":10}),
                                                     memory=memory)
    return qa_chain

#image for bot
bot_template = "<img src='https://upload.wikimedia.org/wikipedia/en/5/5b/WALL%C2%B7E_%28character%29.jpg' width=35 height=35 style='border:1px solid black; border-radius: 50%; object-fit: cover;'>"

#image for user
user_template = "<img src='https://static.tvtropes.org/pmwiki/pub/images/linguinei_feom_ratatsaoiulle_9937.png' width=35 height=35 style='border:1px solid black; border-radius: 50%; object-fit: cover;'>"

def process_question(input_query):
    #send user query to qa chain saved in session state memory
    response = st.session_state.conversation({'question': input_query, 'chat_history': st.session_state.chat_history})

    #save chat history to session state memory
    st.session_state.chat_history = response['chat_history']
    
    #iterate through chat history in reverse order
    for question_num, message in enumerate(st.session_state.chat_history[::-1]):
        #append answers to list so that user is above assistant with the most recent text
        if question_num % 2 != 0:
            st.write(f"<div style='text-align: left;'>{user_template}&emsp;<font size='2'><u>User:</u>&nbsp;&nbsp;{message.content}</font></div><br>",
                     unsafe_allow_html=True)
        else:
            st.write(f"<div style='text-align: left;'>{bot_template}&emsp;<font size='2'><u>Bot:</u>&nbsp;&nbsp;{message.content}</font></div><br>",
                     unsafe_allow_html=True)

#start streamlit instance
st.title('PDF ChatBot')

#initialize chat history and conversation in session memory
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
    
#get user query
user_question = st.text_input("What would you like to ask about your pdf?")

#process user question in streamlit session state
if user_question:
    process_question(user_question)
    
#move document upload to a sidebar
with st.sidebar:
    #hugginface api token input
    HUGGINGFACEHUB_API_TOKEN = st.text_input("Please input HuggingFace API Token.", type="password")
    
    #file input to upload files
    uploaded_file = st.file_uploader('Upload pdf here.', accept_multiple_files=True)
    
    #when button pressed
    if st.button('Read.'):
        with st.spinner('Reading'):
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
            #create document store for pdf
            document_store = create_document_store(uploaded_file)

            #create chain for model to process conversation
            st.session_state.conversation = create_chain(document_store)
            st.write('Finished reading.')
