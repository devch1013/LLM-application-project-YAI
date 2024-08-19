import yaml
from pyprojroot import here


class LoadConfig:
    def __init__(self) -> None:
        with open(here("config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.llama_model_name = app_config["llama_model_name"]
        self.temperature = app_config["temperature"]
        self.max_tokens = app_config["max_tokens"]
        self.articles_to_search = app_config["articles_to_search"]
        self.llm_system_role = app_config["llm_system_role"]
        self.llm_format_output = app_config["llm_format_output"]
        self.chunk_size = app_config["chunk_size"]
        self.similarity_top_k = app_config["similarity_top_k"]