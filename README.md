# Chat + MermaidJS Diagram Generator

This Streamlit application allows users to generate and display MermaidJS diagrams through a chat-like interface powered by the Gemini AI model.

## Features

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
