
def text_splitter(chunk_size, chunk_overlap):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    length_function = len,
    )
    
    return text_splitter


def recursive_splitter(chunk_size, chunk_overlap):
    """
    - `RecursiveCharacterTextSplit`은 재귀적으로 문서를 분할합니다. 먼저, `"\n\n"`(줄바꿈)을 기준으로 문서를 분할하고 이렇게 나눈 청크가 여전히 너무 클 경우에 `"\n"`(문장 단위)을 기준으로 문서를 분할합니다. 그렇게 했을 때에도 청크가 충분히 작아지지 않았다면 문장을 단어 단위로 자르게 되지만, 그렇게까지 세부적인 분할은 자주 필요하지 않습니다.

    - 이런 식의 분할 방법은 문장들의 의미를 최대한 보존하는 형태로 분할할 수 있도록 만들고, 그렇기 때문에 다수의 청크를 LLM에 활용함에 있어서 맥락이 유지되도록 하기에 용이합니다.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )
    return text_splitter

def HTML_splitter(pages):
    from langchain_text_splitters import HTMLHeaderTextSplitter
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_header_splits = html_splitter.split_text(html_string)
    return html_header_splits