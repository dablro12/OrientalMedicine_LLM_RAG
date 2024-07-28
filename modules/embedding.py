import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
import os 
from pathlib import Path

from loader import Dataloader
def get_embeddings(use_bge_embedding = True):
    if use_bge_embedding:
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = OpenAIEmbeddings()
    return embeddings

@st.cache_resource(show_spinner="첨부파일을 통해 고객님의 원하시는 정보를 찾고 있어요")
def binary_embed_file(file, user_urls):
    docs = Dataloader(type = 'binary_file', file = file, urls = user_urls, tokenizer='cl100k_base').extract()
    
    embeddings = get_embeddings()
    cache_dir = LocalFileStore(f"{str(Path(os.getcwd()).parent)}/.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    vectorstore = Chroma.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever, vectorstore

@st.cache_resource(show_spinner="첨부파일들을 통해 고객님의 원하시는 정보를 찾고 있어요")
def multi_embed_files(files, user_urls):
    docs = Dataloader(type = 'multi_files', file = file, urls = user_urls, tokenizer='cl100k_base').extract()

    embeddings = get_embeddings(use_bge_embedding = True)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    vectorstore = Chroma.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever, vectorstore

@st.cache_resource(show_spinner="원하시는 보험사에 대한 서류를 찾고 있어요")
def url_embed_file(user_urls):
    docs = Dataloader(type='url', file=None, urls=user_urls, tokenizer='cl100k_base').extract()
    embeddings = get_embeddings()
    cache_dir = LocalFileStore(f"{str(Path(os.getcwd()).parent)}/.cache/embeddings/insurance_urls.bin")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # Vectorstore를 생성합니다.
    vectorstore = Chroma.from_documents(docs, embedding=cached_embeddings)

    # VectorstoreIndexCreator를 올바르게 초기화합니다.
    index_creator = VectorstoreIndexCreator(embedding=embeddings, vectorstore_cls=Chroma)
    index = index_creator.from_documents(docs)

    retriever = vectorstore.as_retriever()
    return retriever, vectorstore


def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

def kkma_tokenize(text):
    return [token for token in kkma.morphs(text)]

def okt_tokenize(text):
    return [token for token in okt.morphs(text)]

def url_ensemble_embed_file(user_urls, main_retrieval_type = 'faiss', sub_retrieval_type = 'kiwi', main_weight = 0.3, sub_weight = 0.7, search_type = 'mmr'):
    """
    한국형 앙상블 리트리버 : FAISS & BM25 with[Kiwi, Kkama, Okt] : default weight 0.3(faiss), 0.7(kiwi), search type mmr 
    """
    docs = Dataloader(type='url', file=None, urls=user_urls, tokenizer='cl100k_base').extract()
    embeddings = get_embeddings()
    cache_dir = LocalFileStore(f"{str(Path(os.getcwd()).parent)}/.cache/embeddings/insurance_urls.bin")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # Vectorstore를 생성합니다.
    vectorstore = Chroma.from_documents(docs, embedding=cached_embeddings)
    
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    bm25 = BM25Retriever.from_documents(docs)

    # Main Retrieval 
    if main_retrieval_type == 'kiwi':
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        main_retrieval = BM25Retriever.from_documents(docs, preprocess_func = kiwi_tokenize)
    elif main_retrieval_type == 'kkma':
        from konlpy.tag import Kkma
        kkma = Kkma()
        main_retrieval = BM25Retriever.from_documents(docs, preprocess_func = kkma_tokenize)
    elif main_retrieval_type == 'okt':
        from konlpy.tag import Okt
        okt = Okt()
        main_retrieval = BM25Retriever.from_documents(docs, preprocess_func = okt_tokenize)
    elif main_retrieval_type == 'faiss':
        main_retrieval = FAISS.from_documents(docs, embedding=cached_embeddings).as_retriever()
        
    
    # Sub Retrieval
    if sub_retrieval_type == 'kiwi':
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        sub_retrieval = BM25Retriever.from_documents(docs, preprocess_func = kiwi_tokenize)
    elif sub_retrieval_type == 'kkma':
        from konlpy.tag import Kkma
        kkma = Kkma()
        sub_retrieval = BM25Retriever.from_documents(docs, preprocess_func = kkma_tokenize)
    elif sub_retrieval_type == 'okt':
        from konlpy.tag import Okt
        okt = Okt()
        sub_retrieval = BM25Retriever.from_documents(docs, preprocess_func = okt_tokenize)
    elif sub_retrieval_type == 'faiss':
        sub_retrieval = FAISS.from_documents(docs, embedding=cached_embeddings).as_retriever()
        
    # 앙상블 리트리버 생성
    retrieval = EnsembleRetriever(
        retriever = [main_retrieval, sub_retrieval], #default faiss, kiwi
        weight = [main_weight, sub_weight], #default 0.3, 0.7
        search_type= search_type
    )

    return retrieval, vectorstore
    
    