from openai import OpenAI
import os 
import sys 
import json
import re
sys.path.append('../utils')
from utils.json_utils import clean_json_string

class gpt:
    def __init__(self, model_name = 'gpt-4o', choices = 1, token_idx = 1, temperature = 0, stream = False):
        self.api_key = os.environ['OPENAI_API_KEY']
        self.client = OpenAI(api_key = self.api_key)
        self.tokens = [1, 250, 1000]
        self.n = choices
        self.hyperparameters = {
            "frequency_penalty": 0, # [-2,2]  양수 값이면 같은 줄이 될 가능성을 낮춤
            "max_token" : self.tokens[token_idx], # 생성할 수 있는 최대 토큰 수, 입력 토큰과 생성 토큰의 총 길이는 컨텍스트 길이에 의해 제한
            "choices" : self.n, # 입력 메시지에 대해 생성할 선택지 수
            "presence_penalty" : 0, # [-2,2] 양수 값이면 새로운 주제에 대해 이야기할 가능성이 높아짐
            "response_format" : {"type" : "json_object"},  #모델이 출력해야할 형식 지정하는 객체 gpt4-1106 및 gpt-3.5-turbo-1106과 호환
            "seed" : 627, # 동일한 결과를 얻기 위해 사용되는 난수 시드
            "temperature" : temperature, #  [0,2] 값이 높을 수록 더욱 무작위 적
            "top_p" : 1, # temperature이랑 비슷한데, top_p는 확률 분포의 상위 일부를 선택하는 방식 : 고정시켜놓기! :temperature랑 같이 바꾸지 않기
            "stream" : stream, # 스트리밍 형식으로 실시간 답변 출력 가능 : 긴 답볍을 생성할 때 유용적 
            "model_name": model_name,
        }
    def content(self, reason_cnts = 1):
        content = '''You are a Korean medical school student taking the Korean Medical Licensing Examinations to become a Korean doctor. 
        Answer the questions. Since the correct number of answers to this problem is %d, please provide exactly %d responses. 
        Elaborate your reasoning before giving the final answer. Think step by step.
        Give the answer in the following JSON format
        {
        "reasoning": "(elaboarate your reasoning here)",
        "answer": (answer by number, please provide exactly %d responses)
        }'''%(reason_cnts, reason_cnts, reason_cnts)
        return content
    
    def prompt(self, example_question, example_answer, q):
        chat_completion = self.client.chat.completions.create(
            model=self.hyperparameters['model_name'],
            messages=[
                {"role": "system", "content": self.content(self.n)},
                # 1-shot
                {"role": "user", "content": example_question},
                {"role": "assistant", "content": example_answer},
                # Question
                {"role": "user", "content": q},
            ],
            n=self.hyperparameters['choices'],
            max_tokens=self.hyperparameters['max_token'],
            temperature=self.hyperparameters['temperature'],
            
            # presence_penalty=self.hyperparameters['presence_penalty'],
            # top_p=self.hyperparameters['top_p'],
            stream=self.hyperparameters['stream'],
        )
        return chat_completion.choices 

    def run(self, example_question, example_answer, q, label):
        answers = self.prompt(example_question, example_answer, q = q)
        for j in range(len(answers)):
            data_dict = clean_json_string(json_string = answers[j].message.content)
            
            return {
                "question" : q,
                "label" : label,
                "answer" : data_dict['answer'],
                "reasoning" : data_dict['reasoning']
            }
            