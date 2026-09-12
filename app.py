from langchain_ollama import ChatOllama
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from html_templates import css, bot_template, user_template



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(
        model="nemotron-3-nano:4b",
        temperature=0
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first.")
        return

    response = st.session_state.conversation.invoke({
        "question": user_question
    })
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    #st.write(response["answer"])


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")


    st.write(css, unsafe_allow_html=True)  # This will apply the CSS styles to the Streamlit app
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Nerd")

    user_question = st.text_input("Ask a question from the given PDFs")
    if user_question:
        handle_user_input(user_question)

    #st.text_input("ask a Question from the given PDFs")

    

    with st.sidebar:
        st.subheader("put your pdfs here")
        pdf_docs = st.file_uploader("Upload your PDF files and click process", type=["pdf","docx","doc"], accept_multiple_files=True)
        if st.button("Process PDFs"):
            st.spinner("Processing PDFs...")
            #raw text
            raw_text = get_pdf_text(pdf_docs)
            #st.write(raw_text)
            #chunking 
            chunks = get_text_chunks(raw_text)
            #st.write(chunks)
            #vector store
            vectorstore = get_vectorstore(chunks)

            #convo chain
            st.session_state.conversation= get_conversation_chain(vectorstore)


    

if __name__ == "__main__":
    main()