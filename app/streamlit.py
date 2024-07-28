import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import sys 
from pathlib import Path

class SNUHMEDISCApp:
    def __init__(self, prompt, examples):
        self.setup_paths()
        self.setup_page()
        self.initialize_session_state()
        self.LANGSERVE_ENDPOINT = "http://localhost:8000/llm/"
        self.prompt = prompt 
        self.examples = examples
        self.RAG_PROMPT_TEMPLATE = self.format_prompt(self.prompt, self.select_examples(examples))
        self.NO_RAG_PROMPT_TEMPLATE = self.no_retreival_prompt(self.prompt, self.select_examples(examples))
        self.USE_BGE_EMBEDDING = True
        self.setup_sidebar()
        self.print_history()

    def setup_paths(self):
        """ 외부 라이브러리 경로 설정 """
        sys.path.append(str(Path(os.getcwd()).parent))
        sys.path.append('modules')
        from embedding import binary_embed_file, multi_embed_files, url_embed_file
        from matching import matching_insurance
        from prompt import select_examples, format_prompt, No_retrieval_format_prompt, RAG_chain, RAG_simple_chain, qa_chain
        # from reference_load import reference_load
        self.binary_embed_file = binary_embed_file
        self.multi_embed_file = multi_embed_files
        self.url_embed_file = url_embed_file
        self.matching_insurance = matching_insurance
        self.select_examples = select_examples
        self.format_prompt = format_prompt
        self.no_retreival_prompt = No_retrieval_format_prompt
        self.RAG_chain = RAG_chain
        self.RAG_simple_chain = RAG_simple_chain
        self.qa_chain = qa_chain
        # self.reference_load = reference_load

    def setup_page(self):
        st.set_page_config(page_title="Local LLM with RAG", page_icon="💬")
        st.title("Local Koream Medicine LLM with RAG")

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                ChatMessage(role="assistant", content="한의학 박사 AI 어시스턴스입니다. 무엇을 도와드릴까요?")
            ]

    def setup_sidebar(self):
        # sidebar 설정
        with st.sidebar:
            self.file, self.user_urls = None, None
            self.retrieval, self.db = None, None
            
            st.write("")
            # self.options = st.multiselect(
            #     "🔧 보험사 설정(복수설정가능)",
            #     ["AIA생명", "MG손해보험", "농협손해보험", "동양생명", "라이나손보", "롯데손해보험", "삼성화재 다이렉트", "우체국보험", "하나생명", "한화생명", "KB라이프", "삼성화재", "삼성생명"]
            # )
            
            # self.user_urls = self.matching_insurance(
            #     user_insurance_li=self.options,
            #     url_json = 'data/insuarance/insuarance_url.json')
            
            self.file = st.file_uploader(
                "🔧 참고 파일 업로드(여러 파일 업로드가능)",
                type=["pdf", "txt", "docx", "csv", "xlsx", "html", "json"],
                accept_multiple_files=True
            )
            # 1. URLS 
            # 2. URLS + Files
            # 3. Files
            # 4. None
            if self.file:
                if len(self.file) == 1:
                    self.file = self.file[0]
                    self.retrieval, self.db = self.binary_embed_file(self.file, self.user_urls)
                else:
                    self.retrieval, self.db = self.multi_embed_file(self.file, self.user_urls)
            elif self.user_urls:
                self.retrieval, self.db = self.url_embed_file(self.user_urls)
            else:
                self.retrieval, self.db = None, None
                
            
            # if self.user_urls and self.file:  # Check if both user_urls and file are not empty or None
            #     if len(self.file) == 1: # 보험사 설정 O / 단일 파일
            #         self.retrieval, self.db = self.binary_embed_file(self.file[0], self.user_urls)
            #     else: # 보험사 설정 O / 다중 파일
            #         self.retrieval, self.db = self.multi_embed_file(self.file, self.user_urls)
            # elif self.user_urls and not self.file: # 보험사 설정 O / 파일 X
            #     self.retrieval, self.db = self.url_embed_file(self.user_urls)
            # elif not self.user_urls and self.file: # 보험사 설정 X / 파일 O
            #     self.retrieval, self.db = self.binary_embed_file(self.file[0], self.user_urls)
            # else:
            #     self.retrieval, self.db = None, None

    def print_history(self):
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(self, role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def handle_user_input(self, user_input):
        """ Main Operation Function """ 
        self.add_history("user", user_input)
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            ollama = RemoteRunnable(self.LANGSERVE_ENDPOINT)
            chat_container = st.empty()
            if self.retrieval is not None: # Retrieval
                print("RAG Chain")
                prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
                rag_chain = self.RAG_chain(retrieval=self.retrieval, format_docs=self.format_docs, prompt=prompt, ollama=ollama, StrOutputParser=StrOutputParser)
                answer = rag_chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))
                self.add_history("ai", "".join(chunks))
            else: # No Retrieval
                print("No RAG Chain")
                prompt = ChatPromptTemplate.from_template(
                    self.NO_RAG_PROMPT_TEMPLATE
                )
                
                chain = prompt | ollama | StrOutputParser()
                answer = chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))
                self.add_history("ai", "".join(chunks))

            #############################################################################################################################
            
    def display_url_references(self, source_documents, k):
        st.markdown("## 참고 자료")
        for idx, reference in enumerate(source_documents):
            if idx >= k:
                break
            st.markdown(f"**출처:** [{reference.metadata['source']}]({reference.metadata['source']})")
            self.add_history("assistant", f"출처: [{reference.metadata['source']}]({reference.metadata['source']})")
    
    ##############################################################################################################################
    
    def run(self):
        if user_input := st.chat_input():
            self.handle_user_input(user_input)

if __name__ == "__main__":
    # app = SNUHMEDISCApp()
    # app.run()
    # 세션 초기화세팅
    if "first_run" not in st.session_state:
        st.session_state.first_run = True
        # Register session state reset callback
        st.session_state["reset"] = False
    
    INSTRUMENT = "당신은 질문에 정확한 대답을 해주는 선생님입니다. 검색된 다음 문맥을 사용해 질문에 출처를 포함하여 답하세요. <Example Answer>의 형식에 맞춰서 대답하세요. 문서에 대한 내용이 아닌 경우 [내용에 포함된 대답을 해주세요]라고 답하세요.\n"
    #Few-Shot
    EXAMPLE = [
        {"Question": "AIA보험에서 실손의료 보험 청구를 위해 필요한 서류는 무엇인가요?", "Answer": "[진료비계산영수증 및 진료비 세부내역서]와 [입퇴원확인서, 진단서 중 택 1]가 필요합니다."},
        {"Question": "하나 생명보험에서 입원시 공통으로 내어야할 서류를 알려주세요", "Answer": "[입퇴원확인서]와 [진단서]가 필요합니다."},
        {"Question": "라이나 생명보험에서 여행자보험 사고에서 공통서류 중 기본 구비해야할 서류?", "Answer": "[보험금 청구서]와 [여권사본 및 여행일정표]와 [청구인 신분증 사본]가 필요합니다."}
    ]

    app = SNUHMEDISCApp(prompt=INSTRUMENT, examples=EXAMPLE)
    app.run()

    # If the reset flag is set, reset the session state
    if st.session_state["reset"]:
        app.reset_session_state()
        st.session_state["reset"] = False
