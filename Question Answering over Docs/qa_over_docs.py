#IMPORTING MODULES
import wikipedia
import torch
import numpy
import langchain
from collections import UserDict
import os
import sentence_transformers
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain import HuggingFaceHub, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 


#Downloading wikipedia content
topics = ['badminton','cricket','football','Rugby league','basketball',
          'table tennis','chess','paintball','marathon','golf',
          'snooker','baseball','handball','lawn tennis','skating',
          'politician','air pollution','gender','credit','heart']

content = []
docs = []
for i in topics:
    ct = wikipedia.page(i).content
    content.append(ct)
    
    for le in range(0,len(ct)-768,768):
        d = UserDict()
        d.page_content = ct[le:le+768]
        d.metadata = {"page":i}
        docs.append(d)
print(len(docs))

text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)


os.environ['HUGGINGFACEHUB_API_TOKEN'] = '' #ADD YOUR HUGGING FACE TOKEN HERE
# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xxl',
    model_kwargs={'temperature':0.2, 'max_length':2048}
)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=hub_llm, chain_type="stuff", retriever=retriever)

query = "how is football different than cricket"
qa.run(query)

for temp in [0.005,0.1,0.5,0.9]:
    hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xxl',
    model_kwargs={'temperature':temp, 'max_length':5120}
)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=hub_llm, chain_type="stuff", retriever=retriever)
    query = "describe cricket"
    print(f'-----Temperature = {temp}--------')
    print(qa.run(query))