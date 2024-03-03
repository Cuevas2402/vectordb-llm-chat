import streamlit as st
import chromadb
import os

from dotenv import load_dotenv

#Langchain
from langchain.indexes import SQLRecordManager, index
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

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

    loader = DirectoryLoader('./pdfs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    docs_list = text_splitter.split_documents(docs)
    
    return docs_list


def insert_docs(record_manager, vector_db, docs):
    
    index(
        docs,
        record_manager,
        vector_db,
        cleanup="incremental",
        source_id_key="source",
    )

def clean_dir():
    directorio = "./pdfs/"
    for archivo in os.listdir(directorio):
        ruta = os.path.join(directorio, archivo)
        if os.path.isfile(ruta_completa):
            os.remove(ruta_completa)
    

def load_info():
    
    db_client = chromadb.HttpClient(host="localhost", port=8000)
    
    if len(db.list_collections()) == 0:

        create_record_manager()
        db_client.create_collection("langchain")
        

    record_manager = get_record_manager()
    
    docs = load_docs()
    
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    vector_db = Chroma(
        client=db_client,
        collection_name="langchain",
        embedding_function=embedding,
    )

    insert_docs(record_manager, vector_db, docs)
    
    clean_dir()

        
def save_files(archivos):
    directorio = "./archivos/"
    os.makedirs(directorio, exist_ok=True)
    for archivo in archivos:
        with open(os.path.join(directorio, archivo.name), "wb") as f:
            f.write(archivo.getbuffer())
    

def  main():
    load_dotenv()
    st.set_page_config(page_title="Chat", page_icon=":books:")


    st.header("Chat")
    user_question = st.text_input("Has una pregunta")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube tus documentos pdf", accept_multiple_files=True)
        if st.button("Cargar documentos"):
            with st.spinner("Processing"):
                save_files(pdf_docs)
                load_info()

                
if __name__ == '__main__':
    main()