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
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain

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

    loader = DirectoryLoader('./pdfs/', glob="./*.pdf", loader_cls=PyPDFLoader)
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

    llm = OpenAI()
    
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    template = '''

        Historial del chat : {chat_history}
        
        Descripción del problema o pregunta de programación:

        {query}
        
        Ejemplo de código relacionado o contexto adicional (opcional):

        {code}

        Instruccion para el modelo:

        Por favor, proporciona orientación y sugerencias para abordar el problema, pero evita dar soluciones directas o código completo. Me gustaría entender el enfoque y las técnicas para resolver el problema en lugar de obtener una solución final.
    
    '''

    prompt = PromptTemplate.from_template(template)

    chain = LLMChain(llm = llm, prompt = prompt)


    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={
                "score_threshold": 0.5,
                "k":2
            }
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        question_generator=chain,
        retriever=retriever,
        memory=memory
    )
    
    return conversation_chain

def handle_userinput(user_input):
    response = st.session_state.conversation(
        {
            'query': user_question['query'],
            'code' : user_question['code'] if user_question['code'] else ""
        }
    )

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message = st.chat_message("user")
            message.write(message)
        else:
            message = st.chat_message("assistant")
            message.write(message)

######## Fin seccion para conversacion ############

def  main():
    load_dotenv()
    st.set_page_config(page_title="Chat", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat")
    
    col1, col2 = st.columns(2)
    user_question = {}
    with col1:
        user_question["query"] = st.text_input("Ingresa tu pregunta")
    with col2:
        user_question["code"] = st.text_input("Ingresa el codigo (Opcional)")
    ask = st.button("Submit", type = "primary")

    if ask:
        with st.spinner("Processing"):
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