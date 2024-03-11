import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data/")
HTML_PAGES_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "pages/")
PREPROCESSED_DATA_CSV_PATH = os.path.join(DATA_FOLDER_PATH, "preprocessed_data.csv")
CHROMA_DB_INSTANCE_PATH = os.path.join(DATA_FOLDER_PATH, "chromadb_instance")
