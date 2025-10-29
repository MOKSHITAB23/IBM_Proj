from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import tempfile
import streamlit as st

load_dotenv()

if 'llm' not in st.session_state:
    apikey = os.getenv("GROQ_API_KEY")
    st.session_state.llm = ChatGroq(model="llama-3.1-8b-instant", api_key=apikey)

if 'prompt' not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Answer the question based on the provided context and conversation history. "
            "If the question is about previous conversation, refer to the chat history. "
            "Context: {context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ])

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

def create_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embedding=st.session_state.embeddings)
    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(llm=st.session_state.llm, prompt=st.session_state.prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    return chain

def ask_question(question, chain):
    chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    st.session_state.memory.save_context(
        {"input": question},
        {"output": response["answer"]}
    )
    return response["answer"]

st.title("PDF QnA Assistant")

file = st.file_uploader("Upload a PDF to start chatting!", type=["pdf"])

if file:
    if 'chain' not in st.session_state or st.session_state.get('file_name') != file.name:
        with st.spinner("Reading PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(file.read())
                path = temp.name
            st.session_state.chain = create_chain(path)
            st.session_state.file_name = file.name
            st.session_state.messages = []
        st.success(f"Input PDF: {file.name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the Input PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner():
                response = ask_question(prompt, st.session_state.chain)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
