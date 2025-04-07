# Smart diagram

This Streamlit application allows users to generate and display MermaidJS diagrams through a chat-like interface powered by the Gemini AI model.
-   **Uses Google Gemini model for AI processing**
-   **Converts text requirements to MermaidJS syntax**
-   **Real-time diagram rendering**
-   **Persistent session state**
-   **Clean separation of chat and svisualization**
-----------------------------------------------------------------
- code gen
- Initialize App: Load environment variables, configure Gemini API, and set up Streamlit layout.
- Sidebar Inputs: Collect OpenAI API key, Git repo URL, and branch name from the user.

- Swagger Input: Allow users to provide Swagger specs via URL, file upload, or direct paste.

- Parse Swagger: Convert Swagger JSON/YAML into a Python dictionary and extract available model names.

- Model Selection: User selects a model (entity) from the parsed Swagger definitions.

- Clone Git Repo: Clone the specified GitHub repository to a temporary local directory.

- Process Repo: Load .java, .xml, .yml, etc., files from the repo using LangChain's TextLoader.

- Build Vector DB: Split repo files into chunks, embed them using Gemini embeddings, and store them in ChromaDB.

- Generate Code: Use LangChain's RetrievalQA + Gemini to generate:

- Model class

- Repository interface

- Service interface/impl

- Controller class

- Feign client

- Display Output: Show all generated Java code components in collapsible sections with syntax highlighting.## Features

----------------------------------------------------------------------
- code analysis
This Streamlit application, "Git Repository Analyzer," clones a Git repository, analyzes its code, and generates comprehensive documentation using Google's Gemini model and related tools. It features:

Input Handling:
Accepts OpenAI API key via sidebar
Takes Git repository URL as input
Core Functionality:
Clones repository to temporary directory
Extracts code files based on common extensions
Processes files into a vector store using Chroma and embeddings
Generates three outputs:
README.md with project details
Mermaid sequence diagram showing component interactions
Mermaid flow diagram showing process flows
UI Components:
Three tabs for displaying generated content
Modal windows for full-screen diagram viewing
Download options for README.md and diagram HTML files
Technology Stack:
Uses LangChain for LLM operations
ChromaDB for vector storage
Google GenerativeAI for embeddings and text generation
MermaidJS for diagram rendering

The app starts with an input form, processes the repository on button click, and presents results in an organized tabbed interface with interactive diagram viewing capabilities, cleaning up temporary files after processing.


--------------------------------------------------------------------
-   **Chat Interface:** Users can input text instructions or requests for diagram generation.
-   **Gemini AI Integration:** The Gemini AI model processes user input and generates MermaidJS code.
-   **MermaidJS Rendering:** The generated MermaidJS code is rendered as an interactive diagram within the application.
-   **Clean Output:** The application extracts only the MermaidJS code from the AI's response, removing any extraneous text.
-   **Clear Input:** The user input text box is cleared after each request.
-   **HTML Rendering:** MermaidJS diagrams are rendered using HTML for optimal display.

## Prerequisites

-   Python 3.6+
-   Streamlit
-   LangChain
-   python-dotenv
-   google-generativeai

## Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install streamlit langchain python-dotenv google-generativeai
    ```

3.  **Set up your Google API key:**

    -   Create a `.env` file in the same directory as your script.
    -   Add your Google API key to the `.env` file:

        ```
        GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
        ```

4.  **Run the Streamlit application:**

    ```bash
    streamlit run your_script_name.py
    ```

    Replace `your_script_name.py` with the actual name of your Python script.

## Usage

1.  Open the Streamlit application in your web browser.
2.  In the "Chat" section, enter your text instructions for generating a MermaidJS diagram.
3.  The Gemini AI model will process your input and generate the diagram.
4.  The generated diagram will be displayed in the "MermaidJS Diagram" section.
5.  The chat history will show your input.
6.  The input text box will be cleared after each request.

## Code Structure

-   `app2.py`: The main Streamlit application script.
-   `.env`: Stores the Google API key.

## Functions

-   `get_gemini_response(question, chat_history)`: Sends the user's question and chat history to the Gemini AI model and returns the response.
-   `extract_mermaid_code(text)`: Extracts the MermaidJS code from the AI's response, removing any surrounding text.
-   `render_mermaid_html(mermaid_code)`: Generates the HTML code to render the MermaidJS diagram.

## Notes

-   Ensure that your Google API key is correctly set up in the `.env` file.
-   The quality of the generated diagrams depends on the clarity of the user's input.
-   You can customize the system prompt to influence the AI's behavior.
-   Adjust the height of the rendered HTML using the `height` parameter in `st.components.v1.html`.
-   The application uses the `gemini-2.0-flash` model for faster response times.

## Example Input

# Git Repository Analyzer

A Streamlit application that analyzes Git repositories and automatically generates documentation using AI.

## Features

- **Repository Analysis**: Clone and analyze any public Git repository
- **README Generation**: Automatically create comprehensive README.md files
- **Sequence Diagrams**: Generate Mermaid sequence diagrams showing component interactions
- **Flow Diagrams**: Create Mermaid flowcharts illustrating process flows

## How It Works

1. The application clones the specified Git repository
2. Code files are extracted and loaded into a document processing pipeline
3. The text is chunked and embedded using OpenAI embeddings
4. A vector database (ChromaDB) stores the embeddings for semantic search
5. LangChain prompts with GPT-4o generate documentation and diagrams
6. Results are displayed in a user-friendly Streamlit interface

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get an OpenAI API key from [OpenAI's platform](https://platform.openai.com)

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Enter your OpenAI API key in the sidebar
3. Paste the URL of a Git repository you want to analyze
4. Click "Analyze Repository" and wait for the results
5. View and download the generated documentation

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For creating chains of LLM operations
- **ChromaDB**: For vector storage of code chunks
- **OpenAI**: For generating documentation and diagrams
- **Mermaid**: For rendering sequence and flow diagrams

## Limitations

- Large repositories may take longer to process
- The quality of documentation depends on the code quality and comments
- OpenAI API usage incurs costs based on token consumption

## Future Improvements

- Add support for private repositories
- Implement custom diagram styling
- Create options for different documentation formats
- Add code quality analysis
- Include test coverage reporting

