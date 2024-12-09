import streamlit as st
import base64

from ollama import embed
from streamlit import session_state

from vectors import Embeddings
from chatbot import ChatbotClass
import time

if 'temp_pdf_file' not in st.session_state:
    st.session_state['temp_pdf_file'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def display(file):
    base64_pdf = base64.b64encode(file.read()).decode('UTF-8')
    pdf_display = (f"<iframe src='data:application/pdf;base64, {base64_pdf}' width='100%' height='70%'"
                   f"type='application/pdf'></iframe>")
    st.markdown(pdf_display, unsafe_allow_html=True)


st.set_page_config(
    page_title="RAG Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("Your personal PDP analyser")
    menu = ['ChatBot RAG', 'MedicalBot']
    choice = st.selectbox("Models:", menu)

if choice == "ChatBot RAG":
    st.header("ChatBot Llama 3")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("🗃️ Upload your PDF")
        uploader_file = st.file_uploader("Upload", type=["pdf"])
        if uploader_file is not None:
            st.success("File is uploaded successfully!")
            st.markdown(f"File name: {uploader_file.name}")
            st.markdown(f"File size: {uploader_file.size} bytes")
            st.markdown(f"File preview: {display(uploader_file)}")
            file_path = "temp.pdf"
            with open(file_path, "wb") as file:
                file.write(uploader_file.getbuffer())
            st.session_state['temp_pdf_file'] = file_path
    with col2:
        st.title("Embeddings")
        checkbox_embeddings = st.checkbox("Create embeddings")
        if checkbox_embeddings:
            try:
                embeddings_manager = Embeddings(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    qdrant_url="http://localhost:6333",
                    connection_name="vector_db"
                )
                with st.spinner("Creating embeddings...."):
                    result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_file'])
                st.success(result)
                if session_state['chatbot_manager'] is None:
                    st.session_state['chatbot_manager'] = ChatbotClass(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        llm_model="Llama3",
                        temperature=0.7,
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url="http://localhost:6333",
                        connection_name="vector_db"
                    )
            except FileNotFoundError as file_error:
                st.error(file_error)
            except ValueError as ve:
                st.error(ve)
            except Exception as e:
                st.error(e)


