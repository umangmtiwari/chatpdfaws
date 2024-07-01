import json
import os
import sys
import boto3
from flask import Flask, request, jsonify, render_template

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Flask app setup
app = Flask(__name__)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock)
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_vectors', methods=['POST'])
def update_vectors():
    docs = data_ingestion()
    get_vector_store(docs)
    return jsonify({"status": "success"})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get('question')
    model_type = data.get('model')

    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    if model_type == "claude":
        llm = get_claude_llm()
    elif model_type == "llama2":
        llm = get_llama2_llm()
    else:
        return jsonify({"error": "Invalid model type"}), 400

    answer = get_response_llm(llm, faiss_index, user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
