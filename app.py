from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import tempfile
import streamlit as st

load_dotenv()


# ---------------- LLM -----------------
if "llm" not in st.session_state:
    apikey = os.getenv("GROQ_API_KEY")
    st.session_state.llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=apikey
    )


# ---------------- Prompt -----------------
if "prompt" not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. "
            "Answer the question based on the context and conversation history. "
            "If the answer is not found, say you don't know.\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ])


# ---------------- Memory -----------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history"
    )


# ---------------- Embeddings -----------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


# ====================================================
#  Create vector + retrieval pipeline
# ====================================================
def create_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300
    )
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        docs,
        embedding=st.session_state.embeddings
    )
    retriever = vectorstore.as_retriever()

    # DEFINE the chain using LCEL
    chain = (
        st.session_state.prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    return retriever, chain


# ====================================================
#  Ask question
# ====================================================
def ask_question(question, retriever, chain):
    chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", [])

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    response = chain.invoke({
        "context": context,
        "input": question,
        "chat_history": chat_history
    })

    # Save message
    st.session_state.memory.save_context(
        {"input": question},
        {"output": response}
    )

    return response


# ====================================================
# Streamlit UI
# ====================================================
st.title("PDF QnA Assistant")

file = st.file_uploader("Upload a PDF to start chatting!", type=["pdf"])

if file:
    if "chain" not in st.session_state or st.session_state.get("file_name") != file.name:
        with st.spinner("Reading PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(file.read())
                path = temp.name

            retriever, chain = create_chain(path)
            st.session_state.retriever = retriever
            st.session_state.chain = chain
            st.session_state.file_name = file.name
            st.session_state.messages = []

        st.success(f"Input PDF: {file.name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner():
                response = ask_question(
                    prompt,
                    st.session_state.retriever,
                    st.session_state.chain
                )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
