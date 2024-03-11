**Technical assignemtn for MLE role at Autodesk**

To run the full pipeline:
- Step 1: download the HTML pagess dataset into *data/pages*.
- Step 2: run *algos/data_preprocessing.py* to convert the HTML pages content to text, clean it then save the results in a csv file under *data/*.
- Step 3: run *algos/chromadb_functions.py* to create a ChromaDB instance and populate it with the pages contents and their embeddings using OpenAI.
- Step 4: run *main.py* to provide an input query, load the db, retrieve the top-k most relevant documents, provide the documents and the query as input to the chain and get the answer.
