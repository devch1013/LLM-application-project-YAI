from unsloth import FastLanguageModel
import torch
import os
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import default_prompt
from dotenv import load_dotenv
from pathlib import Path

# 프로젝트의 루트 디렉토리로 경로 설정
env_path = Path(__file__).resolve().parent.parent / '.env-secret'

# .env-secret 파일 로드
load_dotenv(dotenv_path=env_path)

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True 
alpaca_prompt = default_prompt.llama_instruct_prompt_for_train

instruction = "Answer user's question"
input = "YAI가 뭐야?"
huggingface_model_name = "devch1013/YAILLAMA"
local_save_name = "llama_weights/fine_tune_100_inst"

# 2. Before Training
print("Before Training")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = os.getenv("HF_TOKEN")
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        system=instruction, # instruction
        user=input, # input
        answer="", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)

# 3. Load data
print("Load Data")
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    # instructions = examples["instruction"]
    instructions = ["Answer user's questions"] * len(examples["question"])
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(system=instruction, user=input, answer=output)
        texts.append(text)
        # print(texts)
        # return 
    return { "text" : texts, }
pass
# dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split = "train")
dataset = load_dataset('csv', data_files='/home/elicer/LLM_project_YAI/llm/llama/output.csv', split="train")
print(dataset)

dataset = dataset.map(formatting_prompts_func, batched = True,)
print(dataset)

# # 4. Training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 100, # Set this for 1 full training run.
        # max_steps = 2000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# # 5. After Training
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         instruction, # instruction
#         input, # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")

# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)

# 6. Saving
model.save_pretrained(local_save_name) # Local saving
tokenizer.save_pretrained(local_save_name)





# model.push_to_hub(huggingface_model_name, token = os.getenv("HF_TOKEN")) 
# tokenizer.push_to_hub(huggingface_model_name, token = os.getenv("HF_TOKEN"))

# Merge to 16bit
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))

# # Merge to 4bit
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_4bit", token = os.getenv("HF_TOKEN"))

# # Just LoRA adapters
# if True: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "lora", token = os.getenv("HF_TOKEN"))

# # Save to 8bit Q8_0
# if True: model.save_pretrained_gguf("model", tokenizer,)
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, token = os.getenv("HF_TOKEN"))

# # Save to 16bit GGUF
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "f16", token = os.getenv("HF_TOKEN"))

# # Save to q4_k_m GGUF
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if True: model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "q4_k_m", token = os.getenv("HF_TOKEN"))

# Save to multiple GGUF options - much faster if you want multiple!


# if True:
#     model.push_to_hub_gguf(
#         huggingface_model_name, # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = os.getenv("HF_TOKEN")
#     )