import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # General embeddings from HuggingFace models.
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import LlamaCpp  # For loading transformer models.
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader, CSVLoader
import tempfile # 임시 파일을 생성하기 위한 라이브러리입니다.
import os
from huggingface_hub import hf_hub_download # Hugging Face Hub에서 모델을 다운로드하기 위한 함수입니다.

# PDF 문서로부터 텍스트를 추출하는 함수입니다.
def get_pdf_text(pdf_docs):
    temp_dir = tempfile.TemporaryDirectory() # 임시 디렉토리를 생성합니다.
    temp_filepath = os.path.join(temp_dir.name, pdf_docs.name) # 임시 파일 경로를 생성합니다.
    with open(temp_filepath, "wb") as f:  # 임시 파일을 바이너리 쓰기 모드로 엽니다.
        f.write(pdf_docs.getvalue()) # PDF 문서의 내용을 임시 파일에 씁니다.
    pdf_loader = PyPDFLoader(temp_filepath) # PyPDFLoader를 사용해 PDF를 로드합니다.
    pdf_doc = pdf_loader.load() # 텍스트를 추출합니다.
    return pdf_doc # 추출한 텍스트를 반환합니다.

    
# 문서들을 처리하여 텍스트 청크로 나누는 함수입니다.
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 청크의 크기를 지정합니다.
        chunk_overlap=200,  # 청크 사이의 중복을 지정합니다.
        length_function=len  # 텍스트의 길이를 측정하는 함수를 지정합니다.
    )

    documents = text_splitter.split_documents(documents)  # 문서들을 청크로 나눕니다.
    return documents  # 나눈 청크를 반환합니다.


# 텍스트 청크들로부터 벡터 스토어를 생성하는 함수입니다.
def get_vectorstore(text_chunks):
    # 원하는 임베딩 모델을 로드합니다.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2',
                                       model_kwargs={'device': 'cpu'})  # 임베딩 모델을 설정합니다.
    vectorstore = FAISS.from_documents(text_chunks, embeddings)  # FAISS 벡터 스토어를 생성합니다.
    return vectorstore  # 생성된 벡터 스토어를 반환합니다.


def get_conversation_chain(vectorstore):
    model_name_or_path = 'TheBloke/Llama-2-7B-chat-GGUF'
    model_basename = 'llama-2-7b-chat.Q2_K.gguf'
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

    llm = LlamaCpp(model_path=model_path,
                   n_ctx=4086,
                   input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
                   verbose=True, )
    # 대화 기록을 저장하기 위한 메모리를 생성합니다.
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # 대화 검색 체인을 생성합니다.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain # 생성된 대화 체인을 반환합니다.

# 사용자 입력을 처리하는 함수입니다.
def handle_userinput(user_question):
    print('user_question =>  ', user_question)
    # 대화 체인을 사용하여 사용자 질문에 대한 응답을 생성합니다.
    response = st.session_state.conversation({'question': user_question})
    # 대화 기록을 저장합니다.
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