from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI


llm = ChatOpenAI(
    # base_url="http://sionic.chat:8001/v1",
    # api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
    # model="xionic-ko-llama-3-70b",
    base_url = "http://sionic.tech:28000/v1",
    api_key = "934c4bbc-c384-4bea-af82-1450d7f8128d",
    model="llama-3.1-xionic-ko-70b",
)

# Prompt 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, smart, kind, and efficient AI assistant named '대현'. You always fulfill the user's requests to the best of your ability. You must generate an answer in Korean.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

####################################################################
# llm = OpenAI(
#     base_url = "http://sionic.tech:28000/v1",
#     api_key = "934c4bbc-c384-4bea-af82-1450d7f8128d"
# )

# # Prompt 설정
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful, smart, kind, and efficient AI assistant named '대현'. You always fulfill the user's requests to the best of your ability. You must generate an answer in Korean.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
####################################################################
chain = prompt | llm | StrOutputParser()
