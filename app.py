import streamlit as st
from dotenv import load_dotenv


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
                acum = 0
                for i in range(100):
                    acum += 1
                
if __name__ == '__main__':
    main()