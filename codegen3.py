import streamlit as st
import os
import tempfile
import shutil
from git import Repo
from langchain.vectorstores import Chroma

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from dotenv import load_dotenv
# Load API Key
load_dotenv()

# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# --- UI ---
st.title("Spring Boot Experience API Generator (LLM + Swagger)")

git_url = st.text_input("Enter Git Repository URL")
swagger_url = st.text_input("Enter Swagger/OpenAPI Spec URL")
generate_btn = st.button("Generate Code")

# --- Processing ---
def clone_repo(repo_url):
    repo_path = tempfile.mkdtemp()
    Repo.clone_from(repo_url, repo_path)
    return repo_path

def index_code_to_vectordb(code_dir):
    docs = []
    loader = TextLoader()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for root, _, files in os.walk(code_dir):
        for file in files:
            if file.endswith(".java"):
                path = os.path.join(root, file)
                docs += splitter.split_documents(loader.load(path))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",request_options={"timeout": 600})

    vectordb = Chroma.from_documents(docs, embeddings)
    return vectordb

def generate_code_from_context(swagger_url, vectordb):
    query = f"""
Given this Swagger URL: {swagger_url}, generate a Spring Boot microservice interface using Feign client
and conforming to existing patterns in the given repo context. 
Use interfaceOnly=true pattern and follow naming, package, and architectural style seen in the codebase.
"""
    retriever = vectordb.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    llm = GoogleGenerativeAI(model="gemini-2.0-flash",
                                                     google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.3)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    response = qa_chain.run(input_documents=relevant_docs, question=query)
    return response

# --- Main Flow ---
if generate_btn:
    if not git_url or not swagger_url:
        st.error("Please provide both the Git repo and Swagger URL.")
    else:
        with st.spinner("Cloning repo and analyzing context..."):
            repo_path = clone_repo(git_url)
            vectordb = index_code_to_vectordb(repo_path)

        with st.spinner("Generating code from context..."):
            generated_code = generate_code_from_context(swagger_url, vectordb)

        st.code(generated_code, language='java')
        st.download_button("Download Code as File", data=generated_code, file_name="ExperienceAPI.java", mime="text/x-java-source")
