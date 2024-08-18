from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv(".env-secret")

llm = ChatOpenAI(verbose=True, model="gpt-4o", temperature=1.0)
output_parser = StrOutputParser()

def order_gpt(system_prompt:str, request: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
        ("user", request)]
        )
    
    chain = prompt | llm | output_parser
    
    result = chain.invoke({})
    return result

# system_prompt = """
# The given article describes YAI, an AI club at Yonsei University.
# Create as many Korean Q&A sets as you can based on this article.
# The order of the questions does not necessarily need to follow the order of the article.
# The questions should clearly state what the question is about, and only ask about one content at a time.
# Especially, the answers should be detailed and long.
# The format is as follows
# [
#     {{
#         "question": "질문",
#         "answer": "대답"
#     }},
#     ...
# ]
# """

system_prompt = """
Based on this article, I want you to create as many different sets of Korean Q&As as possible.
Questions should be clear about who they're for, and only one question should contain one thing.
The longer and more detailed the answer, the better.
The format is as follows
[
    {{
        "question": "논문을 어떻게 읽어야하나요?",
        "answer": "꼼꼼하게 읽어봐야합니다."
    }},
    ...
]
"""

def process_and_save_chatgpt_responses(input_file_path, output_path):
    # OpenAI API 키 설정

    # 입력 파일 읽기
    with open(input_file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # ChatGPT API 호출
    result = order_gpt(system_prompt, input_text)
    # print(result)

    with open(output_path, 'w') as f:
        f.write(result)

    print(f"데이터가 {output_path}에 성공적으로 저장되었습니다.")
    
def process_all_files_in_directory(input_dir, output_dir):
    # 디렉토리 내의 모든 파일 처리
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt') and (filename.startswith("YAI_TIPS1") or filename.startswith("YAI_TIPS2") or filename.startswith("회칙")):
            input_file_path = os.path.join(input_dir, filename)
            for i in range(5,10):
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{i}.json")
                
                print(f"처리 중: {filename}")
                process_and_save_chatgpt_responses(input_file_path, output_path)

    
if __name__ == "__main__":
    process_all_files_in_directory("data", "qna_data")