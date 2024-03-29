
        
class Conversation():
    
    def __init__(self, llm, vector_db, embeddings, retriever_kwargs = {} ):
        self.llm = llm
        self.vector_db = vector_db
        self.embeddings = embeddings 
        self.retriever_kwargs = retriever_kwargs
        self.conversation = self.get_conversation_chain()

    
    def get_conversation_chain():

        llm = OpenAI(temperature=0.5)

        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

        db_client = chromadb.HttpClient(host="localhost", port=8000)

        vector_db = Chroma(
            client=db_client,
            collection_name="langchain",
            embedding_function=embeddings,
        )

        TEMPLATE = '''
            {context}
            Historial del chat : {chat_history}
            
            Descripción del problema o pregunta de programación:

            {question}
            
            Instruccion para el modelo:

            Por favor, proporciona orientación y sugerencias para abordar el problema, pero evita dar soluciones directas o código completo. Me gustaría entender el enfoque y las técnicas para resolver el problema en lugar de obtener una solución final.
        
        '''

        CUSTOM_PROMPT = PromptTemplate(input_variables=["context","chat_history", "question"] , template = TEMPLATE)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        retriever = vector_db.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={
                    "score_threshold": 0.5,
                    "k":2
                }
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={ "prompt": CUSTOM_PROMPT}
        )
    
        return conversation_chain
        
        
        