import os
import warnings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import torch

torch.cuda.empty_cache()  # PyTorch의 GPU 메모리 캐시를 비웁니다.

warnings.filterwarnings("ignore")

folder_path = "notion/data/db"
file_path = "notion/data/index.txt"
loader = TextLoader(file_path, encoding='utf-8')
pages = loader.load()
print("file loaded")

# 문서를 문장으로 분리
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(pages)

# 디렉토리 내부의 모든 txt 파일을 순회하며 로드
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            loader = TextLoader(file_path, encoding='utf-8')
            document = loader.load()
            docs += text_splitter.split_documents(document)
            # docs.extend(document)  # 각 파일의 내용을 하나의 문서로 취급하여 리스트에 추가
print("docs ready")
print("docs count :", len(docs))

# 벡터 저장소 경로 설정
vectorstore_path = 'vectorstore1'

# 기존 벡터 저장소 삭제
# if os.path.exists(vectorstore_path):
#     shutil.rmtree(vectorstore_path)

# os.makedirs(vectorstore_path, exist_ok=True)

# 벡터 임베딩 설정 (GPU 사용, 배치 크기 조절)
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
)


print("embedding ready")

# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
vectorstore.persist()
print("Vectorstore created and persisted")

doc_count = vectorstore._collection.count()
print(f"Number of documents in vectorstore: {doc_count}")
