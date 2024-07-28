import random
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA

def select_examples(examples, sample_cnt=3):
    return random.sample(examples, min(len(examples), sample_cnt))

def No_retrieval_format_prompt(instrument, examples):
    
    example_prompts = "\n".join(
        [f"예시 질문: {example['Question']} <--> 예시 답변 : {example['Answer']}\n" for example in examples]
    )
    
    PROMPT_TEMPLATE = """{instrument}
    <Example Answer>
    {example_prompts}
    
    Question: {question} 
    Answer:"""
    
    prompt = PROMPT_TEMPLATE.format(
        instrument=instrument,
        question="{question}",
        example_prompts=example_prompts,
    )
    return prompt


def format_prompt(instrument, examples):
    
    example_prompts = "\n".join(
        [f"예시 질문: {example['Question']} <--> Answer : {example['Answer']}\n" for example in examples]
    )
    
    PROMPT_TEMPLATE = """{instrument}
    <Example Answer>
    {example_prompts}
        
    Question: {question} 
    Context: {context} 
    Answer:"""
    
    prompt = PROMPT_TEMPLATE.format(
        instrument=instrument,
        question="{question}",
        context="{context}",
        example_prompts=example_prompts,
    )
    return prompt
    


def RAG_chain(retrieval, format_docs, prompt, ollama, StrOutputParser):
    """ 기존 RAG Chain Code
        rag_chain = (
        {
            "context": self.retrieval | self.format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | ollama
        | StrOutputParser()
    )
    """
    
    context_step = retrieval | format_docs
    question_step = RunnablePassthrough()

    # 체인 처리 순서를 좀 더 직관적으로 보여주기 위해 변수로 분리
    steps = {
        "question": question_step,
        "context": context_step,
    }

    # 첫 번째 단계: context와 question을 준비
    step1_result = steps | prompt

    # 두 번째 단계: prompt의 결과를 ollama로 처리
    step2_result = step1_result | ollama

    # 세 번째 단계: ollama의 결과를 문자열로 파싱
    final_result = step2_result | StrOutputParser()

    # 최종 체인 결과
    rag_chain = final_result
    return rag_chain


def RAG_simple_chain(retrieval, format_docs, prompt, ollama, StrOutputParser):
    rag_chain = (
        {
            "context": retrieval | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | ollama
        | StrOutputParser()
    )
    print(rag_chain)
    return rag_chain


def qa_chain(ollma, index):
    qa_chain = RetrievalQA.from_chain_type(
        llm = ollma,
        chain_type = "stuff",
        retrieval = index.as_retrieval()
    )
    return qa_chain
