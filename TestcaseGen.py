import streamlit as st
import os
import git
import glob
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini Pro API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embeddings for vector database
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Set up LLM
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)


# Function to clone git repository or pull latest changes
def setup_repository(repo_url, local_dir):
    if os.path.exists(local_dir):
        repo = git.Repo(local_dir)
        origin = repo.remotes.origin
        origin.pull()
        st.success("Repository successfully updated!")
    else:
        git.Repo.clone_from(repo_url, local_dir)
        st.success("Repository successfully cloned!")
    return local_dir


# Function to extract feature files from repository
def extract_feature_files(repo_dir):
    feature_files = []
    for file_path in glob.glob(f"{repo_dir}/**/*.feature", recursive=True):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            feature_files.append({
                "path": file_path,
                "content": content,
                "name": os.path.basename(file_path)
            })
    return feature_files


# Function to extract step definition files
def extract_step_definitions(repo_dir):
    step_files = []
    # Look for JavaScript/TypeScript step definition files
    for ext in ["js", "ts"]:
        for file_path in glob.glob(f"{repo_dir}/**/*steps*.{ext}", recursive=True):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                step_files.append({
                    "path": file_path,
                    "content": content,
                    "name": os.path.basename(file_path)
                })
    return step_files


# Function to extract cypress test files
def extract_cypress_files(repo_dir):
    cypress_files = []
    for test_pattern in ["spec", "test"]:
        for ext in ["js", "ts", "jsx", "tsx"]:
            for file_path in glob.glob(f"{repo_dir}/**/*{test_pattern}.{ext}", recursive=True):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    cypress_files.append({
                        "path": file_path,
                        "content": content,
                        "name": os.path.basename(file_path)
                    })
    return cypress_files


# Function to create vector store from test files
def create_vector_store(feature_files, step_files, cypress_files):
    # Combine all test-related content
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in feature_files + step_files + cypress_files:
        chunks = text_splitter.split_text(file["content"])
        for chunk in chunks:
            documents.append(f"File: {file['path']}\n\n{chunk}")

    # Create vector store
    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store


# Function to analyze feature file and identify gaps
def analyze_feature_file(feature_content, vector_store=None):
    # Prompt to analyze the feature file and identify missing scenarios
    analysis_prompt = PromptTemplate(
        input_variables=["feature_content"],
        template="""
        You are an expert QA engineer specialized in BDD and Cypress testing.

        Analyze the following feature file:

        {feature_content}

        Identify potential missing test scenarios, edge cases, or user paths that are not covered in the current feature file.
        Focus on:
        1. Error cases
        2. Boundary conditions
        3. Alternative user flows
        4. Common edge cases for this type of feature

        Return a JSON array of missing scenarios, with each scenario having:
        - scenario_name: Brief name for the scenario
        - scenario_description: Why this scenario is important to test
        """
    )

    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    analysis_result = analysis_chain.invoke({"feature_content": feature_content})

    return analysis_result["text"]


# Function to generate cucumber steps in first person perspective
def generate_cucumber_steps(feature_content, missing_scenarios, vector_store=None):
    similar_content = ""

    # If vector store is available, search for similar test patterns
    if vector_store:
        similar_tests = vector_store.similarity_search(feature_content, k=5)
        similar_content = "\n\n".join([doc.page_content for doc in similar_tests])

    # Prompt to generate cucumber steps in first person
    cucumber_prompt = PromptTemplate(
        input_variables=["feature_content", "missing_scenarios", "similar_content"],
        template="""
        You are an expert QA engineer specialized in BDD testing with Cucumber.

        Original feature file:
        {feature_content}

        Missing scenarios to implement:
        {missing_scenarios}

        Similar test patterns from the codebase:
        {similar_content}

        Generate additional Cucumber scenarios in first person perspective for the missing test cases.
        Make sure to:
        1. Use "I" statements for all steps (e.g., "Given I am on the login page")
        2. Follow the existing style and pattern in the original feature file
        3. Be specific about the actions and expectations
        4. Include appropriate tags that match the existing tagging pattern

        Return only the Cucumber feature file content.
        """
    )

    cucumber_chain = LLMChain(llm=llm, prompt=cucumber_prompt)
    cucumber_result = cucumber_chain.invoke({
        "feature_content": feature_content,
        "missing_scenarios": missing_scenarios,
        "similar_content": similar_content
    })

    return cucumber_result["text"]


# Function to generate Cypress code for the cucumber steps
def generate_cypress_code(cucumber_steps, vector_store=None):
    similar_content = ""

    # If vector store is available, search for similar implementations
    if vector_store:
        similar_implementations = vector_store.similarity_search(cucumber_steps, k=5)
        similar_content = "\n\n".join([doc.page_content for doc in similar_implementations])

    # Prompt to generate cypress code
    cypress_prompt = PromptTemplate(
        input_variables=["cucumber_steps", "similar_content"],
        template="""
        You are an expert QA automation engineer specialized in Cypress and Cucumber integration.

        New Cucumber steps to implement:
        {cucumber_steps}

        Similar Cypress implementations from the codebase:
        {similar_content}

        Generate the Cypress test code that implements these Cucumber steps.
        Make sure to:
        1. Follow best practices for Cypress testing
        2. Match the coding style and patterns seen in the similar implementations
        3. Use appropriate selectors that are likely to be stable
        4. Add helpful comments for complex steps
        5. Include proper assertions to verify the expected outcomes

        Return only the Cypress code implementation.
        """
    )

    cypress_chain = LLMChain(llm=llm, prompt=cypress_prompt)
    cypress_result = cypress_chain.invoke({
        "cucumber_steps": cucumber_steps,
        "similar_content": similar_content
    })

    return cypress_result["text"]


# Function to save uploaded file to temp directory
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".feature") as f:
        f.write(uploaded_file.getbuffer())
        return f.name


# Streamlit app interface
def main():
    st.set_page_config(page_title="Test Script Generator", layout="wide")

    st.title("BDD Test Script Generator")
    st.markdown("""
    This app helps you generate missing test cases for your features by analyzing existing test coverage or creating new tests from scratch.
    You can either provide a git repository with existing tests or directly input feature content.
    """)

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Direct Feature Input", "Git Repository Analysis"])

    # Direct Feature Input Tab
    with tab1:
        st.subheader("Generate Tests from Feature Description")

        # Input method selection
        input_method = st.radio("Input Method", ["Text Input", "Upload Feature File"], horizontal=True)

        feature_content = None
        feature_name = None

        if input_method == "Text Input":
            feature_content = st.text_area("Enter Feature Description or Content",
                                           height=300,
                                           help="Enter the feature description or Gherkin language content")
            feature_name = "custom_feature"
        else:
            uploaded_file = st.file_uploader("Upload Feature File", type=["feature", "txt"])
            if uploaded_file is not None:
                feature_name = uploaded_file.name.replace(".feature", "").replace(".txt", "")
                feature_path = save_uploaded_file(uploaded_file)
                with open(feature_path, "r") as f:
                    feature_content = f.read()
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        if feature_content and st.button("Generate Test Cases", key="direct_generate"):
            with st.spinner("Analyzing feature and identifying gaps..."):
                # Analyze feature for gaps
                missing_scenarios = analyze_feature_file(feature_content)
                st.subheader("Identified Missing Scenarios")
                st.json(missing_scenarios)

                # Generate cucumber steps
                st.subheader("Generated Cucumber Steps (First Person)")
                cucumber_steps = generate_cucumber_steps(feature_content, missing_scenarios)
                st.code(cucumber_steps, language="gherkin")

                # Generate cypress code
                st.subheader("Generated Cypress Implementation")
                cypress_code = generate_cypress_code(cucumber_steps)
                st.code(cypress_code, language="javascript")

                # Download options
                col1, col2 = st.columns(2)

                # Save files to disk and provide download links
                cucumber_file = f"{feature_name}_additional.feature"
                cypress_file = f"{feature_name}_additional_steps.js"

                with open(cucumber_file, "w") as f:
                    f.write(cucumber_steps)

                with open(cypress_file, "w") as f:
                    f.write(cypress_code)

                with col1:
                    with open(cucumber_file, "rb") as file:
                        st.download_button(
                            label="Download Cucumber Steps",
                            data=file,
                            file_name=cucumber_file,
                            mime="text/plain"
                        )

                with col2:
                    with open(cypress_file, "rb") as file:
                        st.download_button(
                            label="Download Cypress Code",
                            data=file,
                            file_name=cypress_file,
                            mime="text/javascript"
                        )

    # Git Repository Tab
    with tab2:
        st.subheader("Analyze Existing Test Repository")

        # Repository setup
        with st.expander("Repository Setup", expanded=True):
            repo_url = st.text_input("Git Repository URL", placeholder="https://github.com/username/repo.git")
            repo_dir = st.text_input("Local Directory Name", placeholder="my_repo")

            if st.button("Setup Repository"):
                if repo_url and repo_dir:
                    with st.spinner("Setting up repository..."):
                        setup_repository(repo_url, repo_dir)
                else:
                    st.warning("Please provide both repository URL and directory name.")

        # Only show the rest if repository is set up
        if os.path.exists(repo_dir):
            # Extract files
            with st.spinner("Loading test files..."):
                feature_files = extract_feature_files(repo_dir)
                step_files = extract_step_definitions(repo_dir)
                cypress_files = extract_cypress_files(repo_dir)

                # Create vector store
                vector_store = create_vector_store(feature_files, step_files, cypress_files)

            # Feature selection
            if feature_files:
                feature_names = [f["name"] for f in feature_files]
                selected_feature = st.selectbox("Select a feature to analyze", feature_names)

                # Get selected feature content
                selected_feature_content = next((f["content"] for f in feature_files if f["name"] == selected_feature),
                                                None)

                if selected_feature_content and st.button("Analyze Feature"):
                    with st.spinner("Analyzing feature and identifying gaps..."):
                        # Analyze feature for gaps
                        missing_scenarios = analyze_feature_file(selected_feature_content, vector_store)
                        st.subheader("Identified Missing Scenarios")
                        st.json(missing_scenarios)

                        # Generate cucumber steps
                        st.subheader("Generated Cucumber Steps (First Person)")
                        cucumber_steps = generate_cucumber_steps(selected_feature_content, missing_scenarios,
                                                                 vector_store)
                        st.code(cucumber_steps, language="gherkin")

                        # Generate cypress code
                        st.subheader("Generated Cypress Implementation")
                        cypress_code = generate_cypress_code(cucumber_steps, vector_store)
                        st.code(cypress_code, language="javascript")

                        # Download options
                        col1, col2 = st.columns(2)

                        feature_name = selected_feature.replace(".feature", "")

                        # Save files to disk and provide download links
                        cucumber_file = f"{feature_name}_additional.feature"
                        cypress_file = f"{feature_name}_additional_steps.js"

                        with open(cucumber_file, "w") as f:
                            f.write(cucumber_steps)

                        with open(cypress_file, "w") as f:
                            f.write(cypress_code)

                        with col1:
                            with open(cucumber_file, "rb") as file:
                                st.download_button(
                                    label="Download Cucumber Steps",
                                    data=file,
                                    file_name=cucumber_file,
                                    mime="text/plain"
                                )

                        with col2:
                            with open(cypress_file, "rb") as file:
                                st.download_button(
                                    label="Download Cypress Code",
                                    data=file,
                                    file_name=cypress_file,
                                    mime="text/javascript"
                                )
            else:
                st.warning("No feature files found in the repository. Please check the repository structure.")


if __name__ == "__main__":
    main()