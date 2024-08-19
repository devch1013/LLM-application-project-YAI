"""
작성자: 박준우
"""

from llama_index.legacy import VectorStoreIndex, ServiceContext
from typing import List, Dict, Any

from .src.utils.load_config import LoadConfig
from .src.utils.app_utils import load_data
from .src.utils import arxiv_search
# from llm.yainoma import YAINOMA

llm_format_output =   """
  #Citing sources
  After giving your final answer, you will cite your sources the following way:
  'REFERENCES:
    Title of article -> url
    Title of article -> url
    etc...'
"""

class ArxivRAG:
    def __init__(self, llm_model):
        self.service_context = ServiceContext.from_defaults(llm=llm_model.model)
    
    def build_index(self, _docs):
        
        index = VectorStoreIndex.from_documents(_docs, service_context=self.service_context)
        return index

    def get_papers(self, query, search_result_count = 5):
        search_results: List[Dict[Any]] = arxiv_search.scrape_papers(query, search_result_count)
        return search_results

    def generate(self, question, topk = 5):
        docs = self.get_papers(question)
        index = self.build_index(docs)

        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True,
            similarity_top_k = topk
        )

        response = query_engine.query(question + llm_format_output)

        return response

    def process(self, query):
        """
        Arxiv 검색해서 답변 생성
        """
        # answer = "트랜스포머는 신입니다."
        # reference = {
        #     "link": "https://arxiv.com/~~~",
        #     "title": "Attention is all you need"
        # } 출력 예시
        return answer, reference

if __name__ == "__main__":
    # APPCFG = LoadConfig()

    # papers = load_data()
    llm_model = YAINOMA()

    rag_system = ArxivRAG(llm_model = llm_model)

    query = "Tell me about RAG"
    response = rag_system.generate(query)

    print(f"Question: {query} \n Answer: {response}")