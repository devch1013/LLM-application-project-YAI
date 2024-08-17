from unsloth import FastLanguageModel
from transformers import TextStreamer
from llm.llama import default_prompt

max_seq_length = 2048
dtype = None
load_in_4bit = True

class YAINOMA:
    _instance = None

    def __init__(self, model_dir="/home/elicer/llama_train/fine_tune_500_inst"):
        if YAINOMA._instance is not None:
            ## 이미 YAINOMA가 생성되었으면 다시 생성 불가
            raise Exception("Model already initialized!")

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

        YAINOMA._instance = self

    @classmethod
    def get_instance(cls):
        """
        싱글톤 패턴
        첫 초기화 이후에는 get_instance로 모델 가져오기
        """
        if not cls._instance:
            raise Exception("Model not initialized!")
        else:
            return cls._instance

    def extract_answer(self, text):
        return text.split("assistant", 1)[1]

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

    def simpleQA(self, input_text):
        prompt = "Answer to user's question."
        return self.inference(system_prompt=prompt, input_text=input_text)


if __name__ == "__main__":
    model = YAINOMA()

    ## 바로 question_answering 해도됨

    model2 = YAINOMA.get_instance()

    # questions = ["YAI가 뭐야?", "YAI의 14기 회장이 누구야?", "YAI의 마일리지 제도에 대해 설명해줘"]
    questions = ["제3회 야이콘 우승팀이 누구야?", "YAIVERSE 팀이 무슨 프로젝트를 했는지 알려줘","YAI가 뭐야?", "YAI의 14기 회장이 누구야?", "YAI의 마일리지 제도에 대해 설명해줘"]

    for q in questions:
        print("question: ", q)
        print("answer: ", model2.simpleQA(q))
