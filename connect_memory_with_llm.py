import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Hugging Face Token
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Setup LLM (Mistral with HuggingFace)
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        token=HF_TOKEN,  # FIXED: Token must be passed explicitly
        temperature=0.5,
        model_kwargs={"max_length": 512}  # FIXED: max_length should be an integer
    )
    return llm

# Step 2: Set Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Load FAISS Vector Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS database
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS database loaded successfully.")
except Exception as e:
    print("❌ Error loading FAISS database:", e)
    exit()

# Step 4: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Debug: Check if FAISS is returning results
def test_retrieval(query):
    retrieved_docs = db.similarity_search(query, k=3)
    if retrieved_docs:
        print(f"🔍 {len(retrieved_docs)} documents retrieved.")
        for i, doc in enumerate(retrieved_docs):
            print(f"📄 Document {i+1}: {doc.page_content[:200]}...")  # Print first 200 characters
    else:
        print("⚠️ No relevant documents found in FAISS!")

# Step 5: Handle User Query
user_query = input("Write Query Here: ")

# Debug: Check retrieval results
test_retrieval(user_query)

# Invoke the QA chain
response = qa_chain.invoke({'question': user_query})  # FIXED: "query" → "question"

# Display Results
if response and "result" in response:
    print("\n🎯 RESULT: ", response["result"])
    if response["source_documents"]:
        print("\n📚 SOURCE DOCUMENTS:")
        for doc in response["source_documents"]:
            print(f"📄 {doc.metadata['source']} - {doc.page_content[:200]}...")
else:
    print("❌ No result returned. Try refining your query.")
