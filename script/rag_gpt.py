import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys 
sys.path.append('../')


from langchain_community.callbacks.manager import get_openai_callback
def reply(answer):
    for key, value in answer.items():
        if key == str('source_documents'):
            idx = 0 
            for reference in answer['source_documents']:
                if idx >= 2:
                    break
                print('\n','#'*30, "Reference page", '#'*30)
                print(f"## 참고 내용\n{reference.page_content}")
                print(f"## 내용 출처 : {reference.metadata['source']}, {reference.metadata['page']} 페이지\n")
                
                idx += 1

        else:
            pass 
        
# Baseline : RAG
# if __name__ == "__main__":
    # from src.RAG import RAG

#     RAG_gpt= RAG.RAG(
#         file_path = "/home/eiden/eiden/LLM/langchain/data/insuarance/AIA보험.pdf",
#         chain_type = 'stuff', # "stuff", "map_reduce", "refine", and "map_rerank"
#         streaming = True,
#         search_kwargs = {'k':5, 'fetch_k': 10},
#     )
#     question = "AIA보험에서 실손의료 보험 청구를 위해 필요한 서류는 무엇인가요?"
#     print('#'*30, 'Question', '#'*30)
#     print(question, "\n")
#     print('#'*30, 'Answer', '#'*30)
#     answer = RAG_gpt.run(query = question)
#     reply(answer)

# v0.0.1 : RAG + Dynamic Few-Shot Prompting
# if __name__ == "__main__":
    # from src.RAG_v0_0_1 import RAG

#     # Example usage
#     examples = [
#         {"question": "AIA보험에서 실손의료 보험 청구를 위해 필요한 서류는 무엇인가요?", "answer": "진료비계산영수증 및 진료비 세부내역서 / 입퇴원확인서, 진단서 중 택 1"},
#         {"question": "하나 생명보험에서 입원시 공통으로 내어야할 서류를 알려주세요", "answer": "입퇴원확인서 / 진단서"},
#         {"question": "라이나 생명보험에서 여행자보험 사고에서 공통서류 중 기본 구비해야할 서류?", "answer": "보험금 청구서 / 여권사본 및 여행일정표 / 청구인 신분증 사본"}
#     ]
    
#     RAG_gpt= RAG(
#         file_path = "/home/eiden/eiden/LLM/langchain/data/insuarance/KB생명.pdf",
#         chain_type = 'stuff', # "stuff", "map_reduce", "refine", and "map_rerank"
#         streaming = True,
#         search_kwargs = {'k':5, 'fetch_k': 10},
#         examples = examples
#     )
#     question = "KB생명에서 골절시 기본 서류는 무엇인가요? 추가로 발급은 어디서 해야하는지도 알려주세요"
#     print('#'*30, 'Question', '#'*30)
#     print(question, "\n")
#     print('#'*30, 'Answer', '#'*30)
#     answer = RAG_gpt.run(query = question)
#     reply(answer)
    

# v0.0.2 : v0.0.1 + HTML Loader Packages
if __name__ == "__main__":
    from src.RAG_v0_0_2 import RAG
    # Example usage
    examples = [
        {"question": "AIA보험에서 실손의료 보험 청구를 위해 필요한 서류는 무엇인가요?", "answer": "진료비계산영수증 및 진료비 세부내역서 / 입퇴원확인서, 진단서 중 택 1"},
        {"question": "하나 생명보험에서 입원시 공통으로 내어야할 서류를 알려주세요", "answer": "입퇴원확인서 / 진단서"},
        {"question": "라이나 생명보험에서 여행자보험 사고에서 공통서류 중 기본 구비해야할 서류?", "answer": "보험금 청구서 / 여권사본 및 여행일정표 / 청구인 신분증 사본"}
    ]
    
    urls = [
        "https://www.aia.co.kr/ko/my-aia/my-aia/insurance-application/insurance-claim.html",
        "https://m.mggeneralins.com/RW191010MM.scp?menuId=MN5105002",
        "https://m.nhfire.co.kr/mhwr/web/html/cmps/bilgdcm/retrieveBilgDcmList.html",
        "https://www.myangel.co.kr/customer/service/propose/WE_CR_PS_04_02_06.jsp",
        "https://www.chubb.com/kr-kr/claims/individual-insurance.html",
        "https://www.lotteins.co.kr/web/C/D/C/cdc_claim_0502.jsp",
        "https://direct.samsungfire.com/claim/PP040202_001.html",
        "https://www.epostlife.go.kr/PYIMRD0002.do",
        "https://www.hanalife.co.kr/csc/accidentInsuranceGuide/accidentInsurancePaymentPaymentDocumentSummaryGuide.do",
    # 제외 -> 막아놓음 # "https://www.hanwhalife.com/static/main/myPage/insurance/accident/document/MY_INAPADC_T40000.jsp",
        "https://www.kblife.co.kr/customer-center/informRequiredDocument.do",
        "https://www.samsungfire.com/claim/P_P03_01_02_009.html",
        "https://www.samsunglife.com/individual/cs/guide/MDP-CURDO010110M/0",
    ]
    RAG_gpt= RAG(
        file_path = urls,
        chain_type = 'stuff', # "stuff", "map_reduce", "refine", and "map_rerank"
        streaming = True,
        search_kwargs = {'k':5, 'fetch_k': 10},
        examples = examples
    )
    question = "KB생명 실손의료비(입원)에 대한 기본 서류?"
    print('#'*30, 'Question', '#'*30)
    print(question, "\n")
    print('#'*30, 'Answer', '#'*30)

    answer = RAG_gpt.run(query = question)
    reply(answer)
        