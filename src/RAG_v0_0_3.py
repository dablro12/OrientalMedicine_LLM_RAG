import os
import tiktoken
import torch

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader, AsyncHtmlLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.callbacks.manager import get_openai_callback  # Updated import
import random

class RAG:
    def __init__(self, file_path:list,
        tokenizer='cl100k_base',  
        embed_model='jhgan/ko-sbert-nli', 
        model_kwargs={'device': torch.device("cuda" if torch.cuda.is_available() else "cpu").type},
        encode_kwargs={'normalize_embeddings': True},
        llm_model='gpt-3.5-turbo',
        temperature=0,
        streaming=False,
        chain_type='stuff',
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10},
        examples=None):

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
        
        self.examples = examples if examples is not None else []

    def tiktoken_len(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def data_loader(self, file_path):
        loader = AsyncHtmlLoader(file_path)
        docs = loader.load()
        html2text = Html2TextTransformer()
        pages = html2text.transform_documents(docs)
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
            streaming=streaming, # 타자로 치는 것처럼 나오기위해 streaming True로
            callbacks=[StreamingStdOutCallbackHandler()],   
            temperature=temperature
        )  # 고정 값이 나오도록 만듬 
        qa = RetrievalQA.from_chain_type(llm=openai_llm,  # openai chatgpt 3.5 turbo
                                        chain_type=chain_type,  # retriever 4종류 중 stuff로 사용해보기  # "stuff", "map_reduce", "refine", and "map_rerank".
                                        retriever=self.docsearch.as_retriever(  # 검색에 대해 어떻게 할지? : 벡터저장소에서 docssearch를 검색기로 사용
                                        search_type=search_type,  # 연관성 있는 문서 중 최대한 다양한 Context를 조합해서 LLM에 던져주는 방식
                                        search_kwargs=search_kwargs),  # 구체적으로 search type을 명시하는 옵션 => k : llm에 넘겨줄때 3개를 넘겨줌, fetch_k : 10개 문서를 가져옴 
                                        return_source_documents=True)  # 참고하는 문서의 source를 알 수 있게끔 함
        return qa

    def select_examples(self, k=3):
        return random.sample(self.examples, min(len(self.examples), k))

    def format_prompt(self, query, examples):
        example_prompts = "\n".join(
            [f"Q: {example['question']}\nA: {example['answer']}" for example in examples]
        )
        return f"{example_prompts}\nQ: {query}\nA:"

    def run(self, query):
        examples = self.select_examples()
        prompt = self.format_prompt(query, examples)
        with get_openai_callback() as cb:
            result = self.qa.invoke({"query": prompt})
            
            
            print("\n", "#"*30, "Token Usage", "#"*30)
            print(f"총 사용된 토큰수: \t\t{cb.total_tokens}")
            print(f"프롬프트에 사용된 토큰수: \t{cb.prompt_tokens}")
            print(f"답변에 사용된 토큰수: \t{cb.completion_tokens}")
            print(f"호출에 청구된 금액(USD): \t${cb.total_cost}")
        return result

