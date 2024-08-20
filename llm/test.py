from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',  # 더 작은 모델
    model_kwargs={'device': 'cuda'},  # GPU에서 수행
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 8
    },
)
vectorstore_path = 'rag/vectorstore'
vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
print("Vectorstore loaded")

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


doc_count = vectorstore._collection.count()
print(f"Number of documents in vectorstore: {doc_count}")

context_docs = retriever.invoke("yai")  # 여기에서 'invoke'를 사용하여 문서 검색

# 검색된 문서 출력 (예시)
for doc in context_docs:
    print(doc.page_content)