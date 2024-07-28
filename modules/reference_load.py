def reference_load(answer):
    for key, value in answer.items():
        if key == 'source_documents':
            idx = 0
            for reference in value:
                if idx >= 2:
                    break
                # print(f"## 참고 내용\n{reference.page_content}")
                # print(f'## 내용 출처 : {reference.metadata["source"]}, {reference.metadata["page"]} 페이지\n')
                idx += 1
        else:
            pass