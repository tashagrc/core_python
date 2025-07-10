from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv('reviews.csv')
# change text to vector (numbers)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# save the vector to disk
db_location = "./chrome_langchain_db"
# vector store will be created if it does not exist
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    # iterasi dokumennya
    for i, row in df.iterrows():
        document = Document(
            page_content=row['Title'] + " " + row['Review'],
            metadata={
                "date": row['Date'],
                "rating": row['Rating']
            },
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# ubah db ke vector
vector_store = Chroma(
    collection_name="reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# make vector usable to llm

# lookup documents
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # berapa review yang dicari
)


'''
+------------------------+                     
|   Your Document Corpus |    <- Your data (PDFs, docs, etc.)
+------------------------+                     
            |                                    
            |  (1) Embed all documents                                
            v                                    
+------------------------+                     
| Vector Database (e.g.,  |     <- Stores all document vectors
| FAISS / Chroma / etc.) |                     
+------------------------+                     
            ^                                    
            |                                    
            |  (2) Embed query                              
            |                                    
+------------------------+                     
|  User Query (e.g.,     |                    
| "What is quantum entanglement?")                 
+------------------------+                     
            |                                    
            v                                    
+------------------------+                     
|  Embedding Model        |    <- Converts query to a vector
| (e.g., mxbai-embed-large) |                  
+------------------------+                     
            |                                    
            v                                    
+------------------------+                     
| Vector Similarity Search|    <- Finds top-k similar docs       
+------------------------+                     
            |                                    
            v                                    
+------------------------+                     
| Retrieved Documents     |    <- These are most relevant     
+------------------------+                     
            |                                    
            v                                    
+------------------------+                     
| Language Model (e.g., GPT) |  <- Generates final answer using context
+------------------------+                     
            |                                    
            v                                    
+------------------------+                     
|       Final Answer       |  <- Grounded, contextual reply to user
+------------------------+                     

'''

