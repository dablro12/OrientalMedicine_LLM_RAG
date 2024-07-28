import os 
from pathlib import Path
import tiktoken
import asyncio
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

class Dataloader:
    def __init__(self, type, file = None, tokenizer = 'cl100k_base', urls = None, chunk_size = 500, chunk_overlap = 50):
        self.type = type
        self.file = file
        self.urls = urls
        self.chunk_size= chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer)  # Tokenizer 
        
        
    def tiktoken_len(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def url_loader(self, user_url):
        loader= AsyncHtmlLoader(user_url)
        docs = loader.load()
        html2text = Html2TextTransformer()
        pages = html2text.transform_documents(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap, length_function=self.tiktoken_len)
        texts = text_splitter.split_documents(pages)
        return texts
    
    def binary_file_loader(self, file):
        file_content = file.read()
        file_path = f"{str(Path(os.getcwd()).parent)}/.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        loader = UnstructuredFileLoader(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            length_function=self.tiktoken_len,
        )
        docs = loader.load_and_split(text_splitter=text_splitter)
        return docs
    
    def multi_file_loader(self, files):
        docs = []
        for file in files:
            file_content = file.read()
            file_path = f"{str(Path(os.getcwd()).parent)}/.cache/files/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file_content)
            loader = UnstructuredFileLoader(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                length_function=self.tiktoken_len,
            )
            docs.extend(loader.load_and_split(text_splitter=text_splitter))
        return docs
        
        
    # 클래스 호출시 실행되게끔
    def extract(self):
        if self.type == 'binary_file':
            return self.binary_file_loader(self.file)  ## url 혼합 추가피룡!!!
        elif self.type == 'multi_files':
            return self.multi_file_loader(self.file) ## url 혼합 추가피룡!!!
        else: #only url 
            return self.url_loader(self.urls) 