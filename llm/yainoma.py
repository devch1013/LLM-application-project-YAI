"""
작성자: 박찬혁
"""

import arxiv
from unsloth import FastLanguageModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import TextStreamer
from .llama import default_prompt
from rag.arxiv.src.utils.calculate import calculate_similarity
from rake_nltk import Rake

max_seq_length = 2048
dtype = None
load_in_4bit = True


class YAINOMA:
    _instance = None

    def __init__(
        self, model_dir="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", paper_rag=None
    ):
        # if YAINOMA._instance is not None:
        #     ## 이미 YAINOMA가 생성되었으면 다시 생성 불가
        #     raise Exception("Model already initialized!")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            # model_name = "devch1013/YAILLAMA",
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        self.text_streamer = TextStreamer(self.tokenizer)
        self.llama_template = default_prompt.llama_instruct_prompt
        self.system_prompt = "Your name is YAIbot. You should answer the user's questions. If it's not a question, but a general inquiry or greeting, you can answer naturally."

        self.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-m3',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )

        self.vectorstore = Chroma(persist_directory="rag/vectorstore", embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 7})


        # self.arxivRAG = ArxivRAG(self.model)

        # YAINOMA._instance = self

    # @classmethod
    # def get_instance(cls):
    #     """
    #     싱글톤 패턴
    #     첫 초기화 이후에는 get_instance로 모델 가져오기
    #     """
    #     if not cls._instance:
    #         raise Exception("Model not initialized!")
    #     else:
    #         return cls._instance

    def extract_answer(self, text):
        return text.split("assistant", 1)[1].strip()

    def inference(self, system_prompt, input_text):
        inputs = self.tokenizer(
            [
                self.llama_template.format(
                    system=system_prompt,
                    user=input_text,  # input
                )
            ],
            return_tensors="pt",
        ).to("cuda")

        result = self.model.generate(**inputs, max_new_tokens=1000)

        generated_text = self.tokenizer.decode(result[0], skip_special_tokens=True)
        # print("generated_text: ", generated_text)
        response = self.extract_answer(generated_text)
        return response

    def simple_qa(self, input_text):
        return self.inference(system_prompt=self.system_prompt, input_text=input_text)

    def paper_RAG(self, query):
        """
        Arxiv 문서를 검색해서 답변 생성 @박준우
        """
        ## arxiv 문서 검색
        import re

# 영어 단어만 추출
        english_words = re.findall(r'[a-zA-Z]+', query)
        english_text = ' '.join(english_words)
        rake = Rake()
        rake.extract_keywords_from_text(english_text)
        keywords = rake.get_ranked_phrases()
        print("keywords: ", keywords)
        refined_query = " ".join(keywords)

        # keywords = self.inference("주어지는 문장에서 검색에 쓰일 가장 중요한 키워드를 영어로 뽑아줘. 한 단어만 출력해줘", query)
        # print("llm keywords: ", keywords)
        searched_docs = []
        search = arxiv.Search(
            query=refined_query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = list(search.results())

        for i, p in enumerate(papers):
            paper_doc = {
                "url": p.pdf_url,
                "title": p.title, 
                "abstract": p.summary
            }
            searched_docs.append(paper_doc)


        ## query 임베딩
        query_embedding = self.embeddings.embed_query(query)

        best_doc = None
        best_score = 0
        ## 문서 임베딩
        for doc in searched_docs:
            title = doc["title"]
            abstract = doc["abstract"]
            embedding = self.embeddings.embed_query(f"Title: {title}\n\n {abstract}")
            score = calculate_similarity(query_embedding, embedding)
            if score > best_score:
                best_doc = doc
                best_score = score
        ## 1등으로 응답 뽑기
        if best_doc is None:
            return ("검색 결과가 없습니다.", None)
        title = best_doc["title"]
        abstract = best_doc["abstract"]
        system_prompt = f"주어지는 글을 보고 user의 물음에 답변해라.\nTitle: {title}\n\n {abstract}"
        answer = self.inference(system_prompt, query)
        return answer, best_doc

    def general_RAG(self, query):
        """
        vectorstore의 문서로 답변 생성(blog + notion) @고동현, @김덕용
        """
        print("[!] general RAG inference")
        def format_docs(docs):
            return '\n\n'.join([d.page_content for d in docs])
        context_docs = self.retriever.invoke(query)
        context = format_docs(context_docs)
        print(context)

        return self.inference(system_prompt=context, input_text=query)


if __name__ == "__main__":

    # 테스트 방법
    # cd /home/elicer/LLM_project_YAI
    # python -m llm.yainoma

    model = YAINOMA()

    ## 바로 question_answering 해도됨

    model2 = YAINOMA.get_instance()

    # questions = ["YAI가 뭐야?", "YAI의 14기 회장이 누구야?", "YAI의 마일리지 제도에 대해 설명해줘"]
    questions = [
        "제3회 야이콘 우승팀이 누구야?",
        "YAIVERSE 팀이 무슨 프로젝트를 했는지 알려줘",
        "YAI가 뭐야?",
        "YAI의 14기 회장이 누구야?",
        "YAI의 마일리지 제도에 대해 설명해줘",
    ]

    for q in questions:
        print("question: ", q)
        print("answer: ", model2.simple_qa(q))
