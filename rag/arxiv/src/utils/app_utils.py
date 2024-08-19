import os, shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.legacy import VectorStoreIndex, ServiceContext
from llama_index.legacy import SimpleDirectoryReader


def load_data():
    reader = SimpleDirectoryReader(input_dir="src/data", recursive=True)
    docs = reader.load_data()
    return docs


def RAG(_config, _docs):
    tokenizer = AutoTokenizer.from_pretrained(_config.llama_model_name)
    model = AutoModelForCausalLM.from_pretrained(_config.llama_model_name)

    class LLaMA3ServiceContext(ServiceContext):
        def __init__(self, model, tokenizer, chunk_size):
            self.model = model
            self.tokenizer = tokenizer
            self.chunk_size = chunk_size
            self.callback_manager = ServiceContext.callback_manager


        def generate_response(self, prompt, max_tokens, temperature):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    service_context = LLaMA3ServiceContext(
        model=model,
        tokenizer=tokenizer,
        chunk_size=_config.chunk_size
    )

    index = VectorStoreIndex.from_documents(_docs, service_context=service_context)

    return index


def delete_data():
    print("Cleaning the data folder")
    folder = "src/data"
    for filename in os.listdir(folder):
        if filename != ".gitignore":
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))