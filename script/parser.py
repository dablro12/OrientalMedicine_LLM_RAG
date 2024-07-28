import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os 

def initialize_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return browser

def parse_section(browser, header_xpath, subsections):
    section_data = {}
    section_header = browser.find_element(By.XPATH, header_xpath)
    if section_header:
        section_data['title'] = section_header.text.strip()
        section_data['subsections'] = []
        for subsection in subsections:
            subheader = browser.find_element(By.XPATH, subsection['header'])
            if subheader:
                sub_data = {'title': subheader.text.strip(), 'details': []}
                for detail in subsection['details']:
                    detail_elements = browser.find_elements(By.XPATH, detail)
                    for detail_element in detail_elements:
                        sub_data['details'].append(detail_element.text.strip())
                section_data['subsections'].append(sub_data)
    return section_data

def main(url, parser_selector, save_dir):
    browser = initialize_browser()
    browser.get(url)
    result_html = browser.page_source
    soup = BeautifulSoup(result_html, 'html.parser')

    with open(parser_selector, 'r', encoding='utf-8') as f:
        sections = json.load(f)["sections"]

    parsed_data = {}
    for section_name, section_info in sections.items():
        parsed_data[section_name] = parse_section(
            browser,
            header_xpath=section_info['header_xpath'],
            subsections=section_info['subsections']
        )
    save_path = os.path.join(save_dir, parser_selector.split('/')[-1])
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=4)

    print(f"데이터가 {save_path}로 파일로 저장되었습니다.")
    browser.quit()

if __name__ == "__main__":
    main(
        url = "https://www.hi.co.kr/serviceAction.do?menuId=100632",
        parser_selector = "/home/eiden/eiden/LLM/langchain/data/parser_selector/HD_insurance.json",
        save_dir = "/home/eiden/eiden/LLM/langchain/data/save_data"
    )
