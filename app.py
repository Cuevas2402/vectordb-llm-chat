import streamlit as st

from typing import *
import chromadb
import os

from dotenv import load_dotenv

#Langchain
from langchain.indexes import SQLRecordManager, index
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAI

from utils  import Conversation


######## Inicio seccion para cargar archivos ########


# 1. Extraer informacion
# 2. Convertilar a documentos de langchain
# 3. Si no existe un record manager crearlo

def create_record_manager():
    namespace = "chromadb/langchain"

    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    ) 
    
    record_manager.create_schema()

def get_record_manager():
    
    namespace = "chromadb/langchain"

    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    ) 

    return record_manager


def load_docs():

    loader = DirectoryLoader('./archivos/', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    docs_list = text_splitter.split_documents(docs)
    
    return docs_list


def insert_docs(record_manager, vector_db, docs):
    
    log = index(
        docs,
        record_manager,
        vector_db,
        cleanup="incremental",
        source_id_key="source",
    )

    return log

def clean_dir():
    directorio = "./archivos/"
    for archivo in os.listdir(directorio):
        ruta = os.path.join(directorio, archivo)
        if os.path.isfile(ruta):
            os.remove(ruta)
    

def load_info():
    
    db_client = chromadb.HttpClient(host="localhost", port=8000)
    
    if len(db_client.list_collections()) == 0:

        create_record_manager()
        db_client.create_collection("langchain")
        

    record_manager = get_record_manager()
    
    docs = load_docs()
    
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    
    vector_db = Chroma(
        client=db_client,
        collection_name="langchain",
        embedding_function=embeddings,
    )

    log = insert_docs(record_manager, vector_db, docs)
    
    clean_dir()
    
    msg = f"Añadadidos : {log['num_added']}\n Actualizados : {log['num_skipped']}\n Saltados : {log['num_skipped']}\n Eliminados : {log['num_deleted']}"
    
    return msg

        
def save_files(archivos):
    directorio = "./archivos/"
    os.makedirs(directorio, exist_ok=True)
    for archivo in archivos:
        with open(os.path.join(directorio, archivo.name), "wb") as f:
            f.write(archivo.getbuffer())

###### Fin de sección para cargar archivos ########    

###### Inicio sección para el conversacion ###########

def get_conversation_chain():

    llm = OpenAI(temperature=0.5)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    db_client = chromadb.HttpClient(host="localhost", port=8000)

    vector_db = Chroma(
        client=db_client,
        collection_name="langchain",
        embedding_function=embeddings,
    )

    converasation = Conversation(llm , vector_db) 
  
    return converasation 

def handle_userinput(question):
    
    response = st.session_state.conversation.call(question)
    print(response)
    message = st.chat_message("user")
    message.write(response["question"])
    message = st.chat_message("assistant")
    message.write(response["answer"])

######## Fin seccion para conversacion ############

def  main():
    load_dotenv()
    st.set_page_config(page_title="Chat", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None 
    
    st.header("Chat")
    
    query = st.text_input("Ingresa tu pregunta")
    ask = st.button("Submit", type = "primary")

    if ask and query:
        with st.spinner("Processing"):
            handle_userinput(query)
            

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube tus documentos pdf", accept_multiple_files=True)
        if st.button("Cargar documentos"):
            with st.spinner("Processing"):
                save_files(pdf_docs)
                msg = load_info()
                st.session_state.conversation = get_conversation_chain()
                print(msg)

                
if __name__ == '__main__':
    main()