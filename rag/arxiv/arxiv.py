from llama_index.legacy import VectorStoreIndex, ServiceContext
from src.utils.load_config import LoadConfig
from src.utils.app_utils import load_data
from yainoma import YAINOMA

class RAG:
    def __init__(self, papers):
        self.initialize = YAINOMA()
        self.model = YAINOMA.get_instance()
        self.index = self.build_index(papers)
    
    def build_index(self, _docs):
        service_context = ServiceContext.from_defaults(llm=self.model)
        index = VectorStoreIndex.from_documents(_docs, service_context=service_context)
        return index

    def generate(self, _config, question):
        query_engine = self.index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True,
            similarity_top_k = _config.similarity_top_k
        )

        response = query_engine.query(question + APPCFG.llm_format_output)

        return response

if __name__ == "__main__":
    APPCFG = LoadConfig()

    papers = load_data()

    rag_system = RAG(papers)

    query = "Tell me about RAG"
    response = rag_system.generate(APPCFG, query)

    print(f"Question: {query} \n Answer: {response}")