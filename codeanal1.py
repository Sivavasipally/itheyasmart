import streamlit as st
import os
import tempfile
import subprocess
import glob
import shutil
from pathlib import Path
import base64
from langchain.document_loaders import TextLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

# Load API Key
load_dotenv()

# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Set page configuration
st.set_page_config(page_title="Git Repository Analyzer", layout="wide")

# Title and description
st.title("Git Repository Analyzer")
st.markdown("""
This application analyzes Git repositories and generates documentation including:
- README.md files
- Sequence diagrams
- Flow diagrams
""")

# OpenAI API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
os.environ["OPENAI_API_KEY"] = api_key

# Repository URL input
repo_url = st.text_input("Enter the Git repository URL:")


# Function to create HTML with Mermaid diagram
def create_mermaid_html(mermaid_code, title="Diagram"):
    html_content = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>C4 Model Diagram</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }}
        h1 {{ color: #333; text-align: center; }}
        .mermaid {{ display: flex; justify-content: center; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="mermaid">
            {mermaid_code}
        </div>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
"""
    return html_content


# Function to display HTML in an iframe
def display_mermaid_in_iframe(html_content):
    # Encode HTML content to base64
    b64 = base64.b64encode(html_content.encode()).decode()

    # Create iframe HTML
    iframe_html = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="600" style="border:none;"></iframe>'

    return iframe_html


# Function to extract mermaid code from markdown
def extract_mermaid_code(markdown_text):
    if "```mermaid" in markdown_text:
        start_idx = markdown_text.find("```mermaid") + len("```mermaid")
        end_idx = markdown_text.find("```", start_idx)
        return markdown_text[start_idx:end_idx].strip()
    return ""


# Function to clone the repository
def clone_repository(repo_url, temp_dir):
    try:
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error cloning repository: {e.stderr.decode()}")
        return False


# Function to get all code files from repository
def get_code_files(temp_dir):
    code_extensions = ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.html', '.css', '.jsx',
                       '.tsx']
    files = []

    for ext in code_extensions:
        files.extend(glob.glob(f"{temp_dir}/**/*{ext}", recursive=True))

    return files


# Function to load documents from code files
def load_documents(files):
    documents = []
    for file_path in files:
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file_path}: {str(e)}")

    return documents


# Function to create vector store from documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Create a unique persistent directory name
    persist_directory = tempfile.mkdtemp()

    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",request_options={"timeout": 600})
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store, persist_directory


# Function to generate README
def generate_readme(vector_store, repo_name):
    readme_template = """
    You are an expert code analyzer and technical writer. Based on the code repository information provided, 
    generate a comprehensive README.md file for the project.

    Include the following sections:
    1. Project Title and Description
    2. Installation Instructions
    3. Usage Examples
    4. Main Features
    5. Project Structure
    6. Dependencies
    7. Contributing Guidelines
    8. License Information (assume MIT if not specified)

    Analyze the context and ensure the README is tailored to the specific repository.

    Code snippets: {context}

    Repository name: {repo_name}

    README:
    """

    prompt = PromptTemplate(
        input_variables=["context", "repo_name"],
        template=readme_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                 temperature=0.3)  # ChatOpenAI(temperature=0.1, model="gpt-4o")
    chain = LLMChain(llm=llm, prompt=prompt)

    # Get relevant documents for README generation
    docs = vector_store.similarity_search("project description features installation usage", k=10)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate README
    response = chain.run(context=context, repo_name=repo_name)

    return response


# Function to generate sequence diagram
def generate_sequence_diagram(vector_store):
    sequence_diagram_template = """
    You are an expert software architect. Based on the code provided, generate a Mermaid sequence diagram 
    that illustrates the main interaction flows between components in the system.

    Focus on the key interactions and make sure the diagram is clear and readable.

    Code context: {context}

    Generate a Mermaid sequence diagram using the following format:
    ```mermaid
    sequenceDiagram
    [Your sequence diagram here]
    ```

    Mermaid Sequence Diagram:
    """

    prompt = PromptTemplate(
        input_variables=["context"],
        template=sequence_diagram_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                 temperature=0.3)  # ChatOpenAI(temperature=0.1, model="gpt-4o")
    chain = LLMChain(llm=llm, prompt=prompt)

    # Get relevant documents for sequence diagram generation
    docs = vector_store.similarity_search("function call method invoke API request response workflow process", k=15)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate sequence diagram
    response = chain.run(context=context)

    return response


# Function to generate flow diagram
def generate_flow_diagram(vector_store):
    flow_diagram_template = """
    You are an expert software architect. Based on the code provided, generate a Mermaid flowchart 
    that illustrates the main process flows and decision points in the system.

    Focus on the key processes and make sure the diagram is clear and readable.

    Code context: {context}

    Generate a Mermaid flowchart using the following format:
    ```mermaid
    flowchart TD
    [Your flowchart here]
    ```

    Mermaid Flowchart:
    """

    prompt = PromptTemplate(
        input_variables=["context"],
        template=flow_diagram_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                 temperature=0.3)  # ChatOpenAI(temperature=0.1, model="gpt-4o")
    chain = LLMChain(llm=llm, prompt=prompt)

    # Get relevant documents for flow diagram generation
    docs = vector_store.similarity_search("main process workflow logic decision condition loop algorithm", k=15)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate flow diagram
    response = chain.run(context=context)

    return response


# Initialize session state for diagrams
if 'sequence_diagram' not in st.session_state:
    st.session_state.sequence_diagram = None
if 'flow_diagram' not in st.session_state:
    st.session_state.flow_diagram = None
if 'show_sequence_modal' not in st.session_state:
    st.session_state.show_sequence_modal = False
if 'show_flow_modal' not in st.session_state:
    st.session_state.show_flow_modal = False


# Functions to toggle modals
def show_sequence_modal():
    st.session_state.show_sequence_modal = True


def hide_sequence_modal():
    st.session_state.show_sequence_modal = False


def show_flow_modal():
    st.session_state.show_flow_modal = True


def hide_flow_modal():
    st.session_state.show_flow_modal = False


# Main process when the "Analyze Repository" button is clicked
if st.button("Analyze Repository") and repo_url:
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        with st.spinner("Cloning repository and analyzing code..."):
            # Create a temporary directory for the repository
            temp_dir = tempfile.mkdtemp()

            try:
                # Clone the repository
                if clone_repository(repo_url, temp_dir):
                    # Get repo name from URL
                    repo_name = repo_url.split("/")[-1].replace(".git", "")

                    # Get code files
                    code_files = get_code_files(temp_dir)

                    if code_files:
                        st.success(f"Found {len(code_files)} code files in the repository.")

                        # Load documents
                        documents = load_documents(code_files)

                        if documents:
                            # Create vector store
                            vector_store, persist_directory = create_vector_store(documents)

                            # Create tabs for different outputs
                            readme_tab, sequence_tab, flow_tab = st.tabs(["README", "Sequence Diagram", "Flow Diagram"])

                            # Generate and display README
                            with readme_tab:
                                with st.spinner("Generating README..."):
                                    readme_content = generate_readme(vector_store, repo_name)
                                    st.markdown(readme_content)

                                    # Add download button for README
                                    st.download_button(
                                        label="Download README.md",
                                        data=readme_content,
                                        file_name="README.md",
                                        mime="text/markdown"
                                    )

                            # Generate and display sequence diagram
                            with sequence_tab:
                                with st.spinner("Generating Sequence Diagram..."):
                                    sequence_diagram = generate_sequence_diagram(vector_store)
                                    st.markdown(sequence_diagram)

                                    # Store the diagram in session state
                                    st.session_state.sequence_diagram = sequence_diagram

                                    # Add button to view in modal
                                    st.button("View Sequence Diagram", on_click=show_sequence_modal,
                                              key="view_sequence")

                            # Generate and display flow diagram
                            with flow_tab:
                                with st.spinner("Generating Flow Diagram..."):
                                    flow_diagram = generate_flow_diagram(vector_store)
                                    st.markdown(flow_diagram)

                                    # Store the diagram in session state
                                    st.session_state.flow_diagram = flow_diagram

                                    # Add button to view in modal
                                    st.button("View Flow Diagram", on_click=show_flow_modal, key="view_flow")

                            # Clean up the vector store
                            shutil.rmtree(persist_directory, ignore_errors=True)
                        else:
                            st.error("Could not extract text from code files.")
                    else:
                        st.error("No code files found in the repository.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

# Display sequence diagram modal if show_sequence_modal is True
if st.session_state.show_sequence_modal and st.session_state.sequence_diagram:
    # Create a modal dialog
    with st.container():
        st.markdown("### Sequence Diagram")
        st.button("Close", on_click=hide_sequence_modal)

        # Extract mermaid code
        mermaid_code = extract_mermaid_code(st.session_state.sequence_diagram)
        print(mermaid_code)
        if mermaid_code:
            # Create HTML with mermaid diagram
            html_content = create_mermaid_html(mermaid_code, "Sequence Diagram")

            # Display in iframe
            iframe = display_mermaid_in_iframe(html_content)
            st.markdown(iframe, unsafe_allow_html=True)

            # Add download button for HTML
            st.download_button(
                label="Download HTML",
                data=html_content,
                file_name="sequence_diagram.html",
                mime="text/html"
            )
        else:
            st.error("Could not extract Mermaid code from diagram.")

# Display flow diagram modal if show_flow_modal is True
if st.session_state.show_flow_modal and st.session_state.flow_diagram:
    # Create a modal dialog
    with st.container():
        st.markdown("### Flow Diagram")
        st.button("Close", on_click=hide_flow_modal)

        # Extract mermaid code
        mermaid_code = extract_mermaid_code(st.session_state.flow_diagram)

        if mermaid_code:
            # Create HTML with mermaid diagram
            html_content = create_mermaid_html(mermaid_code, "Flow Diagram")

            # Display in iframe
            iframe = display_mermaid_in_iframe(html_content)
            st.markdown(iframe, unsafe_allow_html=True)

            # Add download button for HTML
            st.download_button(
                label="Download HTML",
                data=html_content,
                file_name="flow_diagram.html",
                mime="text/html"
            )
        else:
            st.error("Could not extract Mermaid code from diagram.")

# Sidebar with additional information
st.sidebar.header("Information")
st.sidebar.info("""
This tool uses:
- **LangChain**: For creating chains of LLM operations
- **ChromaDB**: For vector storage of code chunks
- **OpenAI**: For generating documentation and diagrams
- **Mermaid**: For rendering sequence and flow diagrams
""")

# Add requirements at the bottom of the sidebar
st.sidebar.header("Requirements")
st.sidebar.code("""
streamlit==1.27.0
langchain==0.0.267
chromadb==0.4.13
openai==0.27.8
tiktoken==0.4.0
gitpython==3.1.30
""")