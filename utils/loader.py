def URL_docs_loader(url):
    """ Binary URL Loader """
    from langchain_community.document_loaders import WebBaseLoader
    
    data = WebBaseLoader(url = url).load()
    return data

def URL_docs_unstructed_loader(urls):
    """ Multiple URL Loader """
    from langchain_community.document_loaders.url import UnstructuredURLLoader
    
    pages = UnstructuredURLLoader(urls = urls).load()
    return pages

#---------------------------------------------------------------------#
def pdf_docs_loader(pdf_path):
    """ PDF Docs Loader : 한글 인코딩 처리가 우수한 편"""
    from langchain_community.document_loaders import PyPDFLoader
    pages = PyPDFLoader(pdf_path = pdf_path).load_and_split()
    return pages 

def pdf_docs_unstructed_loader(pdf_path):
    """ PDF Docs Unstructed Loader : 페이지 안 metadata를 많이 줌 but 속도가 느림"""
    from langchain_community.document_loaders import UnstructuredPDFLoader
    pages = UnstructuredPDFLoader(pdf_path).load()
    return pages

def pdf_docs_speed_loader(pdf_path):
    """ PDF Docs Fitz Loader : 읽기 속도가 빠름 : 실시간 필요할 때"""
    import fitz
    pdf_doc = fitz.open(pdf_path)
    return pages

def pdf_docs_addinfo_loader(pdf_path):
    """ PDF Docs Add Info Loader : 속도가 가장 느리나, Author나 다양한 metadata 정보를 제공"""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            pages.append(text)
    return pages

#---------------------------------------------------------------------#
def word_docs_loader(word_path):
    """ Word Docs Loader """
    from langchain_community.document_loaders import Docx2txtLoader
    data = Docx2txtLoader(word_path).load()
    return data

#---------------------------------------------------------------------#
def csv_docs_loader(csv_path, columns_li):
    from langchain_community.document_loaders.csv_loader import CSVLoader
    data = CSVLoader(file_path=csv_path, csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': columns_li
    }).load()
    return data