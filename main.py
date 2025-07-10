from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a cafe.
Answer the question in 1 line max.
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Ask your question (q to quit): ")
    if question.lower() == 'q':
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })
    print(result)