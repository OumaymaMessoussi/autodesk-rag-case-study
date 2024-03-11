import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from global_variables import DATA_FOLDER_PATH, PREPROCESSED_DATA_CSV_PATH


def create_and_populate_chromadb(db_path: str, embedding_model: str) -> None:
    """This function creates a chromadb instance and populates it with the embeddings of the preprocessed data.
    The db instance is then saved locally.

    :param db_path: folder path where the db instance will be persisted.
    :param embedding_model: model to be used to generate the data embeddings.
    :return:
    """
    pages_data = pd.read_csv(PREPROCESSED_DATA_CSV_PATH).head(500)
    pages_data["content"].fillna("", inplace=True)
    embedding = OpenAIEmbeddings(model=embedding_model)

    text = []
    for row in pages_data.to_dict("records"):
        text.append(Document(page_content=row["content"], metadata={"source": row["filename"]}))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, chunk_overlap=200, separators=["\n\n", "\n", " ", "", ".", ",", "!", "?"]
    )
    documents = text_splitter.split_documents(text)
    db = Chroma.from_documents(documents, embedding=embedding, persist_directory=db_path)
    db.persist()


def load_chromadb_instance(db_path: str, embedding_model: str) -> Chroma:
    """This function loads a chromadb collection from a persisted instance locally.

    :param db_path: path where the instance is saved.
    :param embedding_model: model used to generate embeddings.
    :return:
    """
    embedding = OpenAIEmbeddings(model=embedding_model)
    db = Chroma(persist_directory=db_path, embedding_function=embedding)
    return db


if __name__ == "__main__":
    embedding_model = "text-embedding-ada-002"
    create_and_populate_chromadb(DATA_FOLDER_PATH, embedding_model=embedding_model)
