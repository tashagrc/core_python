Tutorial:
https://www.youtube.com/watch?v=E4l91XKQSgw

Create an LLM wrapper using Ollama, a small LLM, and a CSV database so that users can ask questions to a file of CSV data about cafe recommendations.

Things learned:
- Ollama is a tool to simplify the running process of LLM locally on our computer.
- LangChain is a framework that simplifies the development of LLM applications, and it supports general use cases.
- RAG (retrieval augmented generation) is a framework that allows LLM to access information from an external source instead of relying from the knowledge it is trained on.

The flow of the app:
- LLM read document corpus (csv, docs, etc)
- Use OllamaEmbedding to convert the data to a vector and store all document vectors using Chroma
- When a user queries something using natural language, the query gets converted to a  vector
- Retrieve the document, perform vector similarity search to find top-k similar data in docs
- Use a language model (small LLM) to generate the final answer
  
