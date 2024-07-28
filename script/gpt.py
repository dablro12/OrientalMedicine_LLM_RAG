import sys 
sys.path.append('./src')
from prompt import gpt

if __name__ == "__main__":
    gpt_inst = gpt(model_name= 'gpt-4o', choices = 1, temperature = 0.2)

    answer = gpt_inst.run(
        example_question = '''감기가 걸리면 무슨 약을 먹어야 하는가?
        1. 감기약 2. 소화약''',
        example_answer = "1",
        q= '''배아플때는 무슨 약을 먹어야 하는가?
        1. 감기약 2. 소화약''',
        label = "2"
    )
    
    print(answer)