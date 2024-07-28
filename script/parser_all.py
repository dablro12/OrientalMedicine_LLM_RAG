import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import re
import json
# def extract_korean(text):
#     korean_text = re.findall(r'[가-힣]', text)
#     return ''.join(korean_text)
def extract_korean_with_spaces(text):
    korean_text = re.findall(r'[가-힣\s]', text)
    return ''.join(korean_text)

def initialize_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return browser


def parse_medical_expense_1(html_content, tag_id, output_file):
    soup = BeautifulSoup(html_content, 'html.parser')
    tree = {}

    div_sc_req_rel = soup.find('div', id=tag_id)
    if div_sc_req_rel:
        h3_tag = div_sc_req_rel.find('h3')
        if h3_tag:
            tree[h3_tag.get_text(strip=True)] = []

            h4_tag = div_sc_req_rel.find_all('h4')
            doc_info_box_tag = div_sc_req_rel.find_all('div', class_='doc_info_box')
            if h4_tag and doc_info_box_tag:
                detail_li = []
                for idx, (h4, doc_info) in enumerate(zip(h4_tag, doc_info_box_tag)):
                    doc_info_li_tag = doc_info.find_all('li')
                    doc_info_li_texts = [doc.get_text(strip=True) for doc in doc_info_li_tag]
                    doc_info_p_tag = doc_info.find_all('p')
                    doc_info_p_texts = [doc.get_text(strip=True) for doc in doc_info_p_tag]

                    # h4.contents와 h4.contents[1].contents[1] 부분을 안전하게 처리
                    if len(h4.contents) > 1 and hasattr(h4.contents[1], 'contents') and len(h4.contents[1].contents) > 1:
                        src_text = h4.contents[1].contents[1].text
                    else:
                        src_text = None

                    detail_element = {
                        h4.contents[0]: {
                            'src': src_text,  # 발급처 : 발급처명
                            "docs": doc_info_li_texts,  # 세부서류 및 세부내용
                        }
                    }

                    try:
                        ref_text = doc_info_p_tag[0].get_text(strip=True)
                        detail_element[h4.contents[0]]["ref"] = [ref_text]  # 참고사항
                    except (IndexError, AttributeError):
                        detail_element[h4.contents[0]]["ref"] = None

                    detail_li.append(detail_element)

                tree[h3_tag.get_text(strip=True)] = detail_li

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(tree, f, ensure_ascii=False, indent=2)

def parse_medical_expense_2(html_content, tag_id, output_file):
    soup = BeautifulSoup(html_content, 'html.parser')
    tree = {}
    
    div_sc_req_rel = soup.find('div', id=tag_id)
    if div_sc_req_rel:
        h3_tag = div_sc_req_rel.find('h3')
        if h3_tag:
            tree[h3_tag.get_text(strip=True)] = []
            
            h4_tag = div_sc_req_rel.find_all('h4')
            doc_info_box_tag = div_sc_req_rel.find_all('div', class_='doc_info_box')
            if h4_tag and doc_info_box_tag:
                detail_li = []
                for idx, (h4, doc_info) in enumerate(zip(h4_tag, doc_info_box_tag)):
                    doc_info_li_tag = doc_info.find_all('li')
                    doc_info_li_texts = [doc.get_text(strip=True) for doc in doc_info_li_tag]
                    doc_info_p_tag = doc_info.find_all('p')
                    doc_info_p_texts = [doc.get_text(strip=True) for doc in doc_info_p_tag]
                    
                    detail_element = {
                        h4.contents[0]: {
                            'src': h4.contents[1].contents[1].text,  # 발급처 : 발급처명
                            "docs": doc_info_li_texts,  # 세부서류 및 세부내용
                        }
                    }

                    try:
                        ref_text = doc_info_p_tag[0].get_text(strip=True)
                        detail_element[h4.contents[0]]["ref"] = [ref_text]  # 참고사항
                    except (IndexError, AttributeError):
                        detail_element[h4.contents[0]]["ref"] = None
                    
                    detail_li.append(detail_element)
                    
                tree[h3_tag.get_text(strip=True)] = detail_li
                # json 파일로 저장
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(tree, f, ensure_ascii=False, indent=2)
                    
                    
def process_sections(browser, result_html, sections, key, parse_func):
    for tag_id in sections:
        save_path = os.path.join('/home/eiden/eiden/LLM/langchain/data/json', key + '_' + tag_id + '.json')
        print(save_path)
        parse_func(result_html, tag_id, save_path)

def main():
    save_dir = '/home/eiden/eiden/LLM/langchain/data/json'
    os.makedirs(save_dir, exist_ok=True)
    
    with open('/home/eiden/eiden/LLM/langchain/data/parser/tree_selection.json', 'r', encoding='utf-8') as f:
        tree_selection = json.load(f)

    for main_key, sub_key in tree_selection.items():
        parse_func = parse_medical_expense_1 if main_key == 'part1' else parse_medical_expense_2
        
        for key, value in sub_key.items():
            browser = initialize_browser()
            browser.get(value['link'])
            result_html = browser.page_source
            process_sections(browser, result_html, value['sections'], key, parse_func)

if __name__ == "__main__":
    main()