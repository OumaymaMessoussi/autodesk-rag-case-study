from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate

from algos.chromadb_functions import load_chromadb_instance
from global_variables import DATA_FOLDER_PATH


def main_qa(query: str, embedding_model: str, topk: int = 10) -> str:
    """main function to invoke the langchain chat model on an input query and return the answer string.

    :param query: input question.
    :param embedding_model: a string specifying the name of the embedding model used.
    :param topk: number of relevant documents to retrieve.
    :return: a string with the generated answer to the query.
    """
    db = load_chromadb_instance(DATA_FOLDER_PATH, embedding_model)
    retriever = db.as_retriever()
    context = retriever.get_relevant_documents(query=query, k=topk)

    chat = ChatOpenAI(temperature=0)

    template = """
    You are a chatbot tp answer questions based on the context below. If you can't 
    answer strictly using the provided context, reply "I don't know".
    Think step-by-step and give a detailed and correct answer.

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    parser = StrOutputParser()

    chain = prompt | chat | parser

    return chain.invoke({"context": context, "question": query})


if __name__ == "__main__":
    results = main_qa("What does Fusion 360 do?", "text-embedding-ada-002")
    print(results)
