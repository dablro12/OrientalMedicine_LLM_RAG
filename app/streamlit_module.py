import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import sys 
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append('modules')
from embedding import get_embeddings, binary_embed_file, multi_embed_files
# ⭐️ Embedding 설정
# USE_BGE_EMBEDDING = True 로 설정시 HuggingFace BAAI/bge-m3 임베딩 사용 (2.7GB 다운로드 시간 걸릴 수 있습니다)
# USE_BGE_EMBEDDING = False 로 설정시 OpenAIEmbeddings 사용 (OPENAI_API_KEY 입력 필요. 과금)
USE_BGE_EMBEDDING = True

# ⭐️ LangServe 모델 설정(EndPoint)
# 1) REMOTE 접속: 본인의 REMOTE LANGSERVE 주소 입력
# LANGSERVE_ENDPOINT = "https://warm-newly-stag.ngrok-free.app/llm/" # ngrok + lanserve
LANGSERVE_ENDPOINT = "http://localhost:8000/llm/"

# 필수 디렉토리 생성 @Mineru
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 프롬프트를 자유롭게 수정해 보세요!
RAG_PROMPT_TEMPLATE = """당신은 질문에 정확히 답변하는 AI 입니다. 검색된 다음 문맥을 사용해 질문에 출처를 포함하여 답하세요. 답을 모른다면 모른다고 답변하세요. 
Question: {question} 
Context: {context} 
Answer:"""

st.set_page_config(page_title="main1", page_icon="💬")
st.title("SNUH-MEDISC Eiden Local LLM Dev")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="고객님의 보험 서류를 준비해드리는 AI 어시스턴스입니다. 무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

with st.sidebar: # 여러 체크박스를 추가할 수 있는 기능
    st.write("")
    options = st.multiselect(
        "🔧 보험사 설정(복수설정가능)",
        [
            "AIA생명","MG손해보험","농협손해보험","동양생명","라이나손보","롯데손해보험","삼성화재 다이렉트","우체국보험","하나생명","한화생명","KB라이프","삼성화재","삼성생명"
        ]
    ) # -> list 
    
    # 
    file = st.file_uploader(
        "참고 파일 업로드(여러 파일 업로드가능)",
        type=["pdf", "txt", "docx", "csv", "xlsx", "html", "json"],
        accept_multiple_files=True
    )
print(file)
if len(file) > 0:
    if len(file) == 1:
        retriever = binary_embed_file(file[0])
    elif len(file) > 1:
        retriever = multi_embed_file(file)
else:
    file = None

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT) #<-- GPU 서버에서 실행을 위해 ngrok + langserve 이용

        chat_container = st.empty()
        if file is not None:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            # 체인을 생성합니다.
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            answer = rag_chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
        else: # 파일과 보험사 설정이 없는 경우 전체에 대해서 처리
            print(f"file is none")
            prompt = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )
            # 체인을 생성합니다.
            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
