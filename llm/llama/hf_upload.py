from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import HfApi
import os

# 프로젝트의 루트 디렉토리로 경로 설정
env_path = Path(__file__).resolve().parent.parent.parent / '.env_secret'
print(env_path)
# .env-secret 파일 로드
load_dotenv(dotenv_path=env_path)

print(os.getenv("HF_TOKEN"))


api = HfApi()

# 로컬 폴더에 있는 모든 콘텐츠를 원격 Space에 업로드 합니다.
# 파일은 기본적으로 리포지토리의 루트 디렉토리에 업로드 됩니다.
api.upload_folder(
    folder_path="llama_weights/fine_tune_200_inst",
    repo_id="devch1013/YAI-NOMA",
    repo_type="model",
)