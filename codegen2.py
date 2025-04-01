import streamlit as st
import os
import tempfile
import json
import requests
import git
import yaml
from langchain.document_loaders import TextLoader
#from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
#from langchain.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import re
import google.generativeai as genai
from dotenv import load_dotenv
# Load API Key
load_dotenv()

# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Set page title and layout
st.set_page_config(
    page_title="Spring Boot Code Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to parse Swagger JSON/YAML
def parse_swagger(swagger_content, is_yaml=False):
    try:
        if is_yaml:
            return yaml.safe_load(swagger_content)
        else:
            return json.loads(swagger_content)
    except Exception as e:
        st.error(f"Error parsing Swagger: {str(e)}")
        return None


# Function to clone Git repository
def clone_repository(repo_url, branch="main"):
    try:
        temp_dir = tempfile.mkdtemp()
        repo = git.Repo.clone_from(repo_url, temp_dir, branch=branch)
        return temp_dir
    except Exception as e:
        st.error(f"Error cloning repository: {str(e)}")
        return None


# Function to load and process code files
def process_repository(repo_path):
    documents = []

    java_extensions = ['.java', '.kt']
    xml_extensions = ['.xml', '.properties', '.yml', '.yaml']

    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, extension = os.path.splitext(file)

            if extension in java_extensions or extension in xml_extensions:
                try:
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

    return documents


# Function to create vector database
def create_vector_db(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",request_options={"timeout": 600})

    db = Chroma.from_documents(texts, embeddings)
    return db


# Function to generate model class
def generate_model_class(llm, db, model_name, properties):
    # Create a prompt for generating a model class
    prompt = f"""
    Generate a Java Spring Boot model class named {model_name} with the following properties:
    {properties}

    The class should include:
    1. All appropriate annotations
    2. Constructors
    3. Getters and setters
    4. ToString, equals, and hashCode methods

    Follow the coding style from the repository.
    """

    # Use the vector database to retrieve relevant context
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    result = qa_chain.run(prompt)
    return result


# Function to generate repository interface
def generate_repository(llm, db, model_name):
    # Create a prompt for generating a repository interface
    prompt = f"""
    Generate a Spring Data JPA repository interface for a model class named {model_name}.
    Include appropriate annotations and method signatures based on common operations.
    Follow the coding style from the repository.
    """

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    result = qa_chain.run(prompt)
    return result


# Function to generate service class
def generate_service(llm, db, model_name):
    # Create a prompt for generating a service class
    prompt = f"""
    Generate a Spring Boot service class for a model named {model_name}.
    Include:
    1. Service interface with method declarations
    2. Service implementation with appropriate annotations
    3. Business logic for CRUD operations
    4. Any necessary exception handling

    Follow the coding style from the repository.
    """

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    result = qa_chain.run(prompt)
    return result


# Function to generate controller class
def generate_controller(llm, db, model_name, api_paths):
    # Create a prompt for generating a controller class
    path_descriptions = json.dumps(api_paths, indent=2)

    prompt = f"""
    Generate a Spring Boot REST controller for {model_name} with endpoints matching these Swagger paths:
    {path_descriptions}

    Include:
    1. Appropriate annotations (RestController, RequestMapping, etc.)
    2. Methods for each endpoint with proper HTTP method annotations
    3. Parameter handling
    4. Response entity construction
    5. Error handling

    Follow the coding style from the repository.
    """

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    result = qa_chain.run(prompt)
    return result


# Function to generate Feign client
def generate_feign_client(llm, db, model_name, api_paths):
    # Create a prompt for generating a Feign client
    path_descriptions = json.dumps(api_paths, indent=2)

    prompt = f"""
    Generate a Spring Cloud Feign client interface for {model_name} with methods matching these Swagger paths:
    {path_descriptions}

    Include:
    1. @FeignClient annotation with appropriate parameters
    2. Method declarations with proper HTTP method annotations
    3. Parameter annotations
    4. Return types

    Follow the coding style from the repository.
    """

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    result = qa_chain.run(prompt)
    return result


# Function to extract paths for a particular model from Swagger
def extract_paths_for_model(swagger_data, model_name):
    paths = {}

    if 'paths' in swagger_data:
        for path, methods in swagger_data['paths'].items():
            if model_name.lower() in path.lower():
                paths[path] = methods

    return paths


# Function to extract model properties from Swagger
def extract_model_properties(swagger_data, model_name):
    properties = {}

    if 'definitions' in swagger_data:
        for def_name, definition in swagger_data['definitions'].items():
            if def_name.lower() == model_name.lower():
                if 'properties' in definition:
                    properties = definition['properties']
                break

    # If not found in definitions, check components/schemas (OpenAPI 3.0)
    elif 'components' in swagger_data and 'schemas' in swagger_data['components']:
        for schema_name, schema in swagger_data['components']['schemas'].items():
            if schema_name.lower() == model_name.lower():
                if 'properties' in schema:
                    properties = schema['properties']
                break

    return properties


# Streamlit UI
st.title("Spring Boot Microservice Code Generator")
st.markdown("Generate Spring Boot code from Swagger specifications with context from your existing codebase.")

# Sidebar for API key input
with st.sidebar:
    st.header("API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    st.header("Repository Settings")
    git_repo_url = st.text_input("Git Repository URL")
    branch_name = st.text_input("Branch Name", value="main")

# Main panel
col1, col2 = st.columns(2)

with col1:
    st.header("Swagger Input")
    swagger_option = st.radio("Choose input method:", ["URL", "Upload File", "Direct Input"])

    swagger_data = None

    if swagger_option == "URL":
        swagger_url = st.text_input("Swagger URL")
        if swagger_url and st.button("Fetch Swagger"):
            try:
                response = requests.get(swagger_url)
                content_type = response.headers.get('Content-Type', '')

                if 'yaml' in content_type or 'yml' in content_type:
                    swagger_data = parse_swagger(response.text, is_yaml=True)
                else:
                    swagger_data = parse_swagger(response.text)

                if swagger_data:
                    st.success("Swagger specification loaded successfully!")
            except Exception as e:
                st.error(f"Error fetching Swagger: {str(e)}")

    elif swagger_option == "Upload File":
        uploaded_file = st.file_uploader("Upload Swagger file", type=["json", "yaml", "yml"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode()
            is_yaml = uploaded_file.name.endswith(('.yaml', '.yml'))
            swagger_data = parse_swagger(content, is_yaml=is_yaml)

            if swagger_data:
                st.success("Swagger specification loaded successfully!")

    elif swagger_option == "Direct Input":
        input_format = st.radio("Format:", ["JSON", "YAML"])
        swagger_content = st.text_area("Paste Swagger content", height=300)

        if swagger_content and st.button("Parse Swagger"):
            is_yaml = input_format == "YAML"
            swagger_data = parse_swagger(swagger_content, is_yaml=is_yaml)

            if swagger_data:
                st.success("Swagger specification parsed successfully!")

    # Display available models from Swagger
    if swagger_data:
        st.subheader("Available Models")
        models = []

        # Check definitions (Swagger 2.0)
        if 'definitions' in swagger_data:
            models.extend(list(swagger_data['definitions'].keys()))

        # Check components/schemas (OpenAPI 3.0)
        elif 'components' in swagger_data and 'schemas' in swagger_data['components']:
            models.extend(list(swagger_data['components']['schemas'].keys()))

        if models:
            selected_model = st.selectbox("Select a model to generate code for:", models)
        else:
            st.warning("No models found in the Swagger specification.")

with col2:
    st.header("Code Generation")

    if swagger_data and 'selected_model' in locals() :#and openai_api_key and git_repo_url:
        if st.button("Generate Code"):
            with st.spinner("Cloning repository..."):
                repo_path = clone_repository(git_repo_url, branch_name)

            if repo_path:
                with st.spinner("Processing repository files..."):
                    documents = process_repository(repo_path)

                    if documents:
                        with st.spinner("Creating vector database..."):
                            db = create_vector_db(documents)
                            #llm = OpenAI(temperature=0.1)
                            # Initialize LLM
                            llm = GoogleGenerativeAI(model="gemini-2.0-flash",
                                                     google_api_key=os.environ["GOOGLE_API_KEY"])

                            # Extract model properties and API paths
                            properties = extract_model_properties(swagger_data, selected_model)
                            api_paths = extract_paths_for_model(swagger_data, selected_model)

                            st.success("Repository processed and ready for code generation!")

                            # Generate code
                            with st.expander("Model Class", expanded=True):
                                with st.spinner("Generating model class..."):
                                    model_code = generate_model_class(llm, db, selected_model,
                                                                      json.dumps(properties, indent=2))
                                    st.code(model_code, language="java")

                            with st.expander("Repository Interface"):
                                with st.spinner("Generating repository interface..."):
                                    repo_code = generate_repository(llm, db, selected_model)
                                    st.code(repo_code, language="java")

                            with st.expander("Service Class"):
                                with st.spinner("Generating service class..."):
                                    service_code = generate_service(llm, db, selected_model)
                                    st.code(service_code, language="java")

                            with st.expander("Controller Class"):
                                with st.spinner("Generating controller class..."):
                                    controller_code = generate_controller(llm, db, selected_model, api_paths)
                                    st.code(controller_code, language="java")

                            with st.expander("Feign Client"):
                                with st.spinner("Generating Feign client..."):
                                    feign_code = generate_feign_client(llm, db, selected_model, api_paths)
                                    st.code(feign_code, language="java")
                    else:
                        st.error("No valid code files found in the repository.")
    else:
        missing = []
        if not openai_api_key:
            missing.append("OpenAI API Key")
        if not git_repo_url:
            missing.append("Git Repository URL")
        if not swagger_data:
            missing.append("Swagger specification")

        if missing:
            st.info(f"Please provide the following to generate code: {', '.join(missing)}")

# Add explanatory information at the bottom
st.markdown("---")
st.markdown("""
### How the Code Generator Works

1. **Fetch Context**: The application clones your Git repository to understand your coding patterns and architecture.
2. **Process Swagger**: It parses the provided Swagger/OpenAPI specification to understand your API structure.
3. **Generate Code**: Using LangChain and a vector database, it generates Spring Boot code that matches your existing codebase style.
4. **Output Components**: The generator creates model classes, repositories, services, controllers, and Feign clients.

This tool helps maintain consistency across your microservices while accelerating development based on your Swagger contracts.
""")