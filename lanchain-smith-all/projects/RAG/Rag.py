from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os


base_url = "http://localhost:11434"
model_name = "llama3.2:1b"

from dotenv import load_dotenv

load_dotenv('/home/prakhar-tiwari/Desktop/myProjects/AI-Learn/LangChain/lanchain-smith-all/.env')

llm = ChatOllama(
  base_url = base_url,
    model = model_name,
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)

# 🧩 Custom prompt
prompt_template = """
You are a helpful assistant. Use the following extracted context from the user's resume to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template.strip()
)

vectorstore=""
retriever=""
question= "what is my name"

def load_and_split_pdf(pdf_path):
  print("[INFO] loading pdf file...")
  loader = PyPDFLoader(pdf_path)
  pages= loader.load_and_split()
  splitter =RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators = ["\n\n", "\n", "."," ",""]
  )
  docs = splitter.split_documents(pages)
  return docs

def create_vector_store(docs):
  print("[INFO] Creating vector store...")
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vectorstore = FAISS.from_documents(docs, embedding=embeddings)
  return vectorstore

#This is default impl of RAG
def setup_qa_chain(vectorstore):
    print("[INFO] Setting up RetrievalQA chain with Ollama...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

#🧠 Build your own custom RAG pipeline

# 🧠 Setup custom RAG pipeline (returns qa_chain and retriever)
# def setup_custom_rag_chain(vectorstore):
#     print("[INFO] Setting up custom RAG chain...")
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     def qa_chain(question):
#         docs = retriever.invoke(question)
#         context = "\n\n".join(doc.page_content for doc in docs)
#         inputs = {"context": context, "question": question}
#         chain = prompt | llm
#         result = chain.invoke(inputs)
#         return result

#     return qa_chain, retriever


# 🚀 Run the standalone function version (not the chat loop)
def run_custom_rag(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    inputs = {"context": context, "question": question}
    chain = prompt | llm
    result = chain.invoke(inputs)
    return result



# 🤖 Ask questions interactively
def ask_questions(qa_chain):
    print("\n[READY] You can now ask questions about your resume.")
    print("Type 'exit' to quit.")
    while True:
        query = input("\n🧠 Your question: ")
        if query.lower() in ['exit', 'quit']:
            break
        answer = qa_chain(query)
        print(f"\n🤖 Answer: {answer}")

# 🚀 Run the full app
if __name__ == "__main__":
    pdf_path = input("📄 Enter the path to your resume PDF: ").strip()

    if not os.path.isfile(pdf_path):
        print("❌ File not found. Please check the path.")
    else:
        docs = load_and_split_pdf(pdf_path)
        vectorstore = create_vector_store(docs)
        qa_chain = setup_qa_chain(vectorstore)
        ask_questions(qa_chain)



        #/home/prakhar-tiwari/Desktop/myProjects/AI-Learn/LangChain/lanchain-smith-all/projects/RAG/Prakhar-Tiwari.pdf