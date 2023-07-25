import json
import os
import urllib
import requests
import time

#import packages needed for openai llm
import openai
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

#start timer
start = time.time()

#load our open ai information
with open('keys/hf.json', 'r') as rd_f:
    data = json.load(rd_f)
    
os.environ['HUGGINGFACEHUB_API_TOKEN'] = data['HUGGINGFACEHUB_API_TOKEN']

def download_file(download_url, pdf_name):
    """
    Download pdf and save locally from http request.
    """
    response = urllib.request.urlopen(download_url)
    file = open(pdf_name, 'wb')
    file.write(response.read())
    file.close()
    print("Completed")
    
#lord of the rings pdf information
pdf = 'docs/LordofTheRings.pdf'
url = 'https://gosafir.com/mag/wp-content/uploads/2019/12/Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf'
if not os.path.exists(pdf):
    download_file(url, pdf)
    
#add pdf to loader object
loader = PyPDFLoader('docs/LordofTheRings.pdf')
#read pages of loader
pages = loader.load() #use page[i].page_content to access page text if needed
print('PDF pages loaded...')

#set chunk and overlap size
chunk_size = 1000
chunk_overlap = 150
#create a text splitter using double new lines, then new lines, then periods, and so on if needed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            separators=['\n\n', '\n', '(?<=\. )', ' ', ''],
                                            length_function=len)
#split pages into chunks using our teext splitter
chunks = text_splitter.split_documents(pages)
print('Pages split...')

#load embeddings
embeddings = HuggingFaceEmbeddings()

#create faiss document store
document_store = FAISS.from_documents(documents=chunks,
                                       embedding=embeddings) #persist_directory lets us save locally
print('Document store created...')

#save database locally
document_store.save_local('LoTR')

#end timer
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Program took {:0>2} hours {:0>2} minutes and {:0>2} seconds to finish".format(int(hours),int(minutes),int(seconds)))