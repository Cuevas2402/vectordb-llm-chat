import streamlit as st

from typing import *
import chromadb
import os

from dotenv import load_dotenv

from langchain_openai import OpenAI

from utils  import Conversation, VectorDB

######## Inicio seccion para cargar archivos ########
        
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

    if "db" not in st.session_state:
        st.session_state.db = VectorDB(namespace = "chromadb/langchain", db_url="sqlite:///record_manager_cache.sql", directory = './archivos/', collection = "langchain" )
    
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
                msg = st.session_state.db.load_info()

                
if __name__ == '__main__':
    main()