import os
import tiktoken
import torch

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback

class RAG:
    def __init__(self, file_path,
        tokenizer='cl100k_base',  
        embed_model='jhgan/ko-sbert-nli', 
        model_kwargs={'device': torch.device("cuda" if torch.cuda.is_available() else "cpu").type},
        encode_kwargs={'normalize_embeddings': True},
        llm_model='gpt-3.5-turbo',
        temperature=0,
        streaming=False,
        chain_type='stuff',
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10}):

        self.tokenizer = tiktoken.get_encoding(tokenizer)  # Tokenizer 
        self.texts = self.data_loader(file_path)  # Data Loader
        self.docsearch = self.embedding(
            texts=self.texts,
            embed_model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.qa = self.build_model(model_name=llm_model, temperature=temperature, streaming=streaming,
                                chain_type=chain_type, search_type=search_type, search_kwargs=search_kwargs)

    def tiktoken_len(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def data_loader(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=self.tiktoken_len)
        texts = text_splitter.split_documents(pages)
        return texts

    def embedding(self, texts, embed_model_name, model_kwargs, encode_kwargs):
        hf = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        docsearch = Chroma.from_documents(texts, hf)
        return docsearch

    def build_model(self, model_name, temperature, streaming, chain_type, search_type, search_kwargs):
        openai_llm = ChatOpenAI(
            model_name=model_name,
            streaming=streaming, callbacks=[StreamingStdOutCallbackHandler()],  # 타자로 치는 것처럼 나오기위해 streaming True로 
            temperature=temperature
        )  # 고정 값이 나오도록 만듬 
        qa = RetrievalQA.from_chain_type(llm=openai_llm,  # openai chatgpt 3.5 turbo
            chain_type=chain_type,  # retriever 4종류 중 stuff로 사용해보기  # "stuff", "map_reduce", "refine", and "map_rerank".
            retriever=self.docsearch.as_retriever(  # 검색에 대해 어떻게 할지? : 벡터저장소에서 docssearch를 검색기로 사용
            search_type=search_type,  # 연관성 있는 문서 중 최대한 다양한 Context를 조합해서 LLM에 던져주는 방식
            search_kwargs=search_kwargs),  # 구체적으로 search type을 명시하는 옵션 => k : llm에 넘겨줄때 3개를 넘겨줌, fetch_k : 10개 문서를 가져옴 
            return_source_documents=True)  # 참고하는 문서의 source를 알 수 있게끔 함
        return qa

    def run(self, query):
        with get_openai_callback() as cb:
            result = self.qa.invoke(query)
            print("#"*30, "Token Usage Viewer", "#"*30)
            print("\n입력 토큰 수:", cb.prompt_tokens)
            print("출력 토큰 수:", cb.completion_tokens)
            print("총 토큰 수:", cb.total_tokens, "\n")
            print("#"*67)
        
        return result