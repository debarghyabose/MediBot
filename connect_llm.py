import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)
##prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use only the context below to answer the question.

Give a clear, concise answer in bullet points.
If the answer is not in the context, say:
"I could not find this information in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt
#vector db
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#query retrieval
qa_chain=RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
 ),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)
#query input
user_query = input("Write Query Here: ")

response = qa_chain.invoke({"query": user_query})

print("\n--- SOURCES ---")

for number, document in enumerate(response["source_documents"], start=1):
    source = document.metadata.get("source", "Unknown source")
    page = document.metadata.get("page", 0) + 1

    print(f"\nSource {number}: {source}")
    print(f"Page: {page}")

print("\n--- ANSWER ---")
print(response["result"])