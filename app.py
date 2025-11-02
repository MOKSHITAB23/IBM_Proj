from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import tempfile
import streamlit as st

load_dotenv()

# Initialize session state variables
if 'llm' not in st.session_state:
    apikey = os.getenv("GROQ_API_KEY")
    st.session_state.llm = ChatGroq(model="llama-3.1-8b-instant", api_key=apikey)

if 'prompt' not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based on the provided context and conversation history. "
                   "If the question is about previous conversation, refer to the chat history. "
                   "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

if 'store' not in st.session_state:
    st.session_state.store = {}

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_chain(pdf_path):
    """Create RAG chain from PDF"""
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=300
    )
    docs = splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(docs, embedding=st.session_state.embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create chains
    document_chain = create_stuff_documents_chain(
        llm=st.session_state.llm, 
        prompt=st.session_state.prompt
    )
    
    retrieval_chain = create_retrieval_chain(
        retriever=retriever, 
        combine_docs_chain=document_chain
    )
    
    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return chain_with_history

def ask_question(question, chain):
    """Ask a question using the RAG chain with history"""
    response = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "pdf_chat_session"}}
    )
    return response["answer"]

# Streamlit UI
st.title("📄 PDF QnA Assistant")

file = st.file_uploader("Upload a PDF to start chatting!", type=["pdf"])

if file:
    # Create or update chain when new file is uploaded
    if 'chain' not in st.session_state or st.session_state.get('file_name') != file.name:
        with st.spinner("Reading PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(file.read())
                path = temp.name
            
            st.session_state.chain = create_chain(path)
            st.session_state.file_name = file.name
            st.session_state.messages = []
            st.session_state.store = {}  # Reset chat history for new PDF
            
            # Clean up temp file
            os.unlink(path)
        
        st.success(f"✅ Loaded PDF: {file.name}")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the PDF..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(prompt, st.session_state.chain)
            st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
