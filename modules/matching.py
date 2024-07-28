import simdjson as json
def matching_insurance(user_insurance_li, url_json):
    url_dict = json.load(open(url_json))
    
    # 보험사 이름과 URL 매칭
    user_urls = []
    for user_insurance in user_insurance_li:
        user_urls.append(url_dict[user_insurance])
    
    if user_urls == []:
        return None
    return user_urls