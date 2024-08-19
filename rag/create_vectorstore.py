import os
import warnings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")

file_path = "blog.txt"
loader = TextLoader(file_path, encoding='utf-8')
pages = loader.load()
print("file loaded")

# 문서를 문장으로 분리
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(pages)
print("docs ready")

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)
print("embedding ready")

# 벡터 저장소 생성
vectorstore = Chroma.from_documents(docs, embeddings)

# 벡터 저장소 경로 설정
vectorstore_path = 'vectorstore'
os.makedirs(vectorstore_path, exist_ok=True)

# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
vectorstore.persist()
print("Vectorstore created and persisted")

