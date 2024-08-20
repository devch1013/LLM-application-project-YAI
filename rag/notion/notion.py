import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from llm.yainoma import YAINOMA
import numpy as np


warnings.filterwarnings("ignore")
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',  # 더 작은 모델
    model_kwargs={'device': 'cuda'},  # GPU에서 수행
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 8
    },
)

vectorstore_path = '../rag/vectorstore'
vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
print("Vectorstore loaded")


retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# YAINOMA 모델 초기화
model = YAINOMA()

# Prompt 템플릿 생성
# template = '''친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.:
# {context}

# Question: {question}
# '''

llama_instruct_prompt = """
<|start_header_id|>system<|end_header_id|>{context}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# prompt = ChatPromptTemplate.from_template(template)
prompt = ChatPromptTemplate.from_template(llama_instruct_prompt)


def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


def calculate_similarity(embedding1, embedding2):
    """임베딩 간 코사인 유사도를 계산."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def generate_answer(question, retriever, model, chat_history, similarity_threshold=0.5):
    """검색된 문서의 유사도가 낮을 때 일반적인 답변 생성."""
    # 1. 질문의 임베딩 생성
    question_embedding = embeddings.embed_query(question)
    
    # 2. 검색된 문서 조각을 통해 문맥 생성
    context_docs = retriever.get_relevant_documents(question)
    context = format_docs(context_docs)
    print(context)
    
    # 3. 검색된 문서와 질문 간의 유사도 계산
    if context_docs:
        doc_embedding = embeddings.embed_query(context_docs[0].page_content)  # 첫 번째 문서 조각과의 유사도 계산
        similarity = calculate_similarity(question_embedding, doc_embedding)
    else:
        similarity = 0  # 검색된 문서가 없으면 유사도를 0으로 설정

    # 4. 유사도가 임계값보다 낮으면 일반적인 답변 생성
    if similarity < similarity_threshold:
        prompt_text = prompt.format(context=chat_history, question=question)
    else:
        # 프롬프트 생성 (이전 대화 내역도 포함)
        print("RAG searching...")
        prompt_text = prompt.format(context=chat_history + context, question=question)
    
    full_prompt =  prompt_text
    # print(full_prompt)

    # 5. 외부 모델을 사용하여 답변 생성
    answer = model.inference(full_prompt, "")
    
    return answer


def chat_loop():
    """여러 번의 대화를 처리하는 루프."""
    chat_history = "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.\n"
    print("챗봇과 대화를 시작합니다. 종료하려면 'exit'를 입력하세요.")
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("대화를 종료합니다.")
            break
        
        answer = generate_answer(query, retriever, model, chat_history)
        
        print(f"Bot: {answer}")
        
        # 대화 히스토리에 추가
        chat_history += f"User: {query}\nBot: {answer}\n\n"


# 챗봇 대화 루프 시작
chat_loop()
