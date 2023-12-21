import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import LlamaCpp  
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
import tempfile
import os
from huggingface_hub import hf_hub_download 

# PDF 문서로부터 텍스트를 추출하는 함수
def get_pdf_text(pdf_docs):
    temp_dir = tempfile.TemporaryDirectory() 
    temp_filepath = os.path.join(temp_dir.name, pdf_docs.name) 
    with open(temp_filepath, "wb") as f:  
        f.write(pdf_docs.getvalue()) 
    pdf_loader = PyPDFLoader(temp_filepath) 
    pdf_doc = pdf_loader.load()
    return pdf_doc 

    
# 문서들을 처리하여 텍스트 청크로 나누는 함수
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200, 
        length_function=len  
    )

    documents = text_splitter.split_documents(documents)  
    return documents 


# 텍스트 청크들로부터 벡터 스토어를 생성하는 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2',
                                       model_kwargs={'device': 'cpu'})  
    vectorstore = FAISS.from_documents(text_chunks, embeddings) 
    return vectorstore 


def get_conversation_chain(vectorstore):
    model_name_or_path = 'TheBloke/Llama-2-7B-chat-GGUF'
    model_basename = 'llama-2-7b-chat.Q2_K.gguf'
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

    llm = LlamaCpp(model_path=model_path,
                   n_ctx=4086,
                   input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
                   verbose=True, )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain 

# 사용자 입력을 처리하는 함수
def handle_userinput(user_question):
    print('user_question =>  ', user_question)
    response = st.session_state.conversation({'question': user_question})
    # 대화 기록을 저장
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Files",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple Files:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                doc_list = []

                for file in docs:
                    print('file - type : ', file.type)
                    if file.type in ['application/octet-stream', 'application/pdf']:
                        # file is .pdf
                        doc_list.extend(get_pdf_text(file))

                # get the text chunks
                text_chunks = get_text_chunks(doc_list)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
