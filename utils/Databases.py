
from langchain.indexes import SQLRecordManager, index
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    
    def __init__(self, namespace, db_url, directory, collection):
        self.db = setup()
        self.directory = DirectoryLoader
        self.collection = collection
        self.record_manager = SQLRecordManager(namespace, db_url) 
    
    def setup():

        try:
         
            db_client = chromadb.HttpClient(host="localhost", port=8000)
            
            self.c_collections = len(db_client.list_collections())
            
            docs = load_docs()
            
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
            
            self.db = Chroma(
                client=db_client,
                collection_name="langchain",
                embedding_function=embeddings,
            )
            
        except:
            
            self.db = None
            
    def load_docs(self):

        loader = DirectoryLoader('./archivos/', glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
        docs_list = text_splitter.split_documents(docs)
        
        return docs_list

    def clean_dir():
        directorio = "./archivos/"
        for archivo in os.listdir(directorio):
            ruta = os.path.join(directorio, archivo)
            if os.path.isfile(ruta):
                os.remove(ruta)
        

    def load_info():
        
        try: 

            if not self.c_collections:

                self.record_manager.create_schema()
                db_client.create_collection("langchain")
            
            docs = load_docs()
            
            log = index(
                docs,
                self.record_manager,
                self.vector_db,
                cleanup="incremental",
                source_id_key="source",
            )

            clean_dir()
            
            msg = f"Añadadidos : {log['num_added']}\n Actualizados : {log['num_skipped']}\n Saltados : {log['num_skipped']}\n Eliminados : {log['num_deleted']}"
            
            return msg
        
        except:
            
            return "Error al cargar los archivos"
