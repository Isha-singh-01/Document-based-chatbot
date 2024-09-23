import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template


def extract_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device':'cpu'})
    vector_db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_db

def get_QA(vector_db):
    llm = CTransformers(
        model = "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        max_new_tokens = 1048,
        temperature = 0.3
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa_prompt_template = PromptTemplate(
        template="Use the provided context to answer the question. If the information is not in the context, say you don't know since its information isn't available in the provided context. Strictly answer on the basis of the pdf document. If any of the definitions or descriptions aren't provided in the pdf, don't answer such questions. Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={'k':2}, search_type="mmr"),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt_template}
    )
    return conversation_chain

def get_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Contextual Q&A Bot", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Contextual Q&A Bot :books:')
    user_question = st.text_input('Ask a question')

    if user_question and st.session_state.conversation:
        get_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents:")
        docs = st.file_uploader("Upload your documents", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing..."):
                # Extract text from PDF
                raw_text = extract_pdf_text(docs)
                
                # Text splitting
                chunks = get_chunks(raw_text)
                
                # Vector DB
                vector_db = get_vector_db(chunks)

                # QA
                st.session_state.conversation = get_QA(vector_db)
            st.success("Processing completed!")

if __name__ == '__main__':
    main()
