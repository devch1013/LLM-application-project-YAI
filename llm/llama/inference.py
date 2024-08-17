from unsloth import FastLanguageModel
import os
from transformers import TextStreamer
import default_prompt

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
alpaca_prompt = default_prompt.llama_instruct_prompt

instruction = "사용자가 물어보는 질문에 답변해줘"

# 2. Before Training
print("Before Training")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/elicer/llama_train/fine_tune_500_inst",
    # model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=os.getenv("HF_TOKEN"),
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
while True:
    input_text = input("Query: ")
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                system = "Answer user's question",  # instruction
                user = input_text,  # input
                # "", # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)
