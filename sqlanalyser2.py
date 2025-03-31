import streamlit as st
import pandas as pd
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document
import sqlparse
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Set page config
st.set_page_config(page_title="SQL Performance Analyzer", layout="wide")

# Add OpenAI API key input
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Application title
st.title("SQL Performance Analyzer")
st.markdown("Upload your SQL schema and queries for performance analysis and optimization suggestions.")

# Database connection section
st.sidebar.header("Database Connection")

# Database type selection
db_type = st.sidebar.selectbox(
    "Select Database Type",
    ["SQLite", "MySQL", "MariaDB", "MSSQL", "Oracle"]
)

# Connection details
if db_type == "SQLite":
    db_path = st.sidebar.text_input("SQLite Database Path", "example.db")
    connection_string = f"sqlite:///{db_path}"
    connect_args = {}
else:
    host = st.sidebar.text_input("Host", "localhost")
    port = st.sidebar.text_input("Port", "3306" if db_type in ["MySQL",
                                                               "MariaDB"] else "1433" if db_type == "MSSQL" else "1521")
    database = st.sidebar.text_input("Database Name")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if db_type == "MySQL":
        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "MariaDB":
        connection_string = f"mariadb+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "MSSQL":
        connection_string = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    elif db_type == "Oracle":
        connection_string = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={database}"

    connect_args = {}

# SQL query input
st.header("Enter your SQL query")
sql_query = st.text_area("SQL Query", height=150)

# Schema upload or input
st.header("Upload or Enter Schema Information")
schema_option = st.radio("Schema Input Method", ["Upload SQL File", "Enter Schema Text"])

if schema_option == "Upload SQL File":
    schema_file = st.file_uploader("Upload Schema SQL File", type=["sql"])
    if schema_file is not None:
        schema_content = schema_file.read().decode("utf-8")
else:
    schema_content = st.text_area("Enter Schema SQL", height=200)

# Initialize connection and analysis
if st.button("Analyze SQL Query"):
    if not sql_query:
        st.error("Please enter a SQL query to analyze.")
    elif not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        try:
            with st.spinner("Connecting to database and analyzing query..."):
                # Connect to the database
                db = SQLDatabase.from_uri(connection_string, **connect_args)

                # Initialize LLM
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                 temperature=0.3)

                # Create a vector store for schema information if provided
                if schema_content:
                    # Parse the schema to extract table and index information
                    st.subheader("Schema Information")
                    statements = sqlparse.split(schema_content)
                    tables_info = []

                    for statement in statements:
                        parsed = sqlparse.parse(statement)[0]
                        if parsed.get_type() == 'CREATE':
                            tables_info.append(statement)

                    if tables_info:
                        st.write(f"Found {len(tables_info)} table definitions in the schema")

                    # Create embeddings for schema info
                    embeddings =  GoogleGenerativeAIEmbeddings(model="models/embedding-001",request_options={"timeout": 600})

                    # Fix: Create proper Document objects with page_content
                    schema_docs = [Document(page_content=info, metadata={"type": "schema"}) for info in tables_info]

                    if schema_docs:
                        schema_db = Chroma.from_documents(
                            schema_docs,
                            embeddings,
                            collection_name="schema_info"
                        )

                # SQL Query Analysis prompt
                analyze_prompt = PromptTemplate(
                    input_variables=["query", "schema", "db_type"],
                    template="""
                    You are an expert database performance tuner specializing in {db_type}.

                    Analyze the following SQL query for performance issues:
                    ```sql
                    {query}
                    ```

                    Schema information:
                    {schema}

                    Provide a comprehensive analysis including:
                    1. Query complexity assessment
                    2. Potential bottlenecks
                    3. Index recommendations
                    4. Join optimizations
                    5. WHERE clause improvements
                    6. Suggest a rewritten, optimized version of the query

                    Be specific about which indexes would help and why.
                    """
                )

                # Create the LLM chain
                analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt)

                # Get schema information from the database
                if hasattr(db, 'get_table_info'):
                    db_schema = db.get_table_info()
                else:
                    # Fallback to provided schema
                    db_schema = schema_content

                # Run the analysis
                analysis_result = analyze_chain.run(
                    query=sql_query,
                    schema=db_schema,
                    db_type=db_type
                )

                # Display the results
                st.subheader("Query Analysis Results")
                st.markdown(analysis_result)

                # Try to execute the query (if safe)
                query_tool = QuerySQLDataBaseTool(db=db)

                try:
                    # First, check if it's a SELECT query (safer to execute)
                    parsed_query = sqlparse.parse(sql_query)[0]
                    if parsed_query.get_type() == 'SELECT':
                        st.subheader("Query Execution Results")
                        with st.spinner("Executing query..."):
                            result = query_tool.run(sql_query)
                            # Convert result to DataFrame if possible
                            try:
                                if isinstance(result, str) and result.strip().startswith(('(', '[')):
                                    import ast

                                    data = ast.literal_eval(result)
                                    df = pd.DataFrame(data)
                                    st.dataframe(df)
                                else:
                                    st.text(result)
                            except:
                                st.text(result)
                    else:
                        st.info("Non-SELECT queries are not executed for safety reasons.")
                except Exception as e:
                    st.warning(f"Could not execute query: {str(e)}")

                # Generate optimization recommendations with more specific database knowledge
                optimize_prompt = PromptTemplate(
                    input_variables=["query", "schema", "db_type", "analysis"],
                    template="""
                    Based on your analysis of this {db_type} query:
                    ```sql
                    {query}
                    ```

                    And the schema:
                    {schema}

                    Previous analysis: {analysis}

                    Provide specific optimization steps for {db_type}:
                    1. Exact index definitions that should be created (with proper syntax)
                    2. Query rewrite with {db_type}-specific optimizations
                    3. Configuration recommendations if applicable
                    4. Execution plan considerations

                    Format your response with clear sections and code examples.
                    """
                )

                optimize_chain = LLMChain(llm=llm, prompt=optimize_prompt)
                optimization_result = optimize_chain.run(
                    query=sql_query,
                    schema=db_schema,
                    db_type=db_type,
                    analysis=analysis_result
                )

                st.subheader("Optimization Recommendations")
                st.markdown(optimization_result)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add a footer with usage instructions
st.markdown("---")
st.markdown("""
### How to use this tool:
1. Enter your OpenAI API key in the sidebar
2. Configure your database connection details
3. Enter your SQL query in the text area
4. Upload a schema file or enter schema information manually
5. Click "Analyze SQL Query" to get performance recommendations
""")