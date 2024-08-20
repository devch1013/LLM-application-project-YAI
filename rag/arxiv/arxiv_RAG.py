"""
작성자: 박준우
"""

from llama_index.legacy import VectorStoreIndex, ServiceContext
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine

from typing import List, Dict, Any

from src.utils.load_config import LoadConfig
from src.utils.app_utils import load_data
from src.utils import arxiv_search
from llm.yainoma import YAINOMA

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

    def generate(self, question, topk = 2):
        docs = self.get_papers(question)
        index = self.build_index(docs)

        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True,
            similarity_top_k = topk
        )

        response = query_engine.query(question + llm_format_output)
    
        reference = response.source_nodes[0].get_text()

        return response, reference

if __name__ == "__main__":
    llm_model = YAINOMA()

    rag_system = ArxivRAG(llm_model = llm_model)

    query = "Tell me about RAG"
    answer, reference = rag_system.generate(query)

    print(f"Question: {query} \n Answer: {answer} \n Reference: {reference}")