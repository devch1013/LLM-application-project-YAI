import numpy as np

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


def calculate_similarity(embedding1, embedding2):
    """임베딩 간 코사인 유사도를 계산."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
