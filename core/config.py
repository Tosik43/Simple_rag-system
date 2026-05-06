from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE")
METADATA_FILE = os.getenv("METADATA_FILE")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWD")

HF_TOKEN= os.getenv("HF_TOKEN")

TOP_K_RETRIEVE = 10
TOP_K_RERANK = 3
MIN_SIMILARITY_SCORE = 0.3

MAX_FILE_SIZE_MB = 10 
MAX_FILES = 10


NO_ANSWER_MESSAGE = "Нет информации по этому вопросу."

LOG_FILE = "rag_logs.csv"