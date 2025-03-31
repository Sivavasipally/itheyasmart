import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load API Key
load_dotenv()

# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(question, chat_history):
    model = genai.GenerativeModel(model_name='gemini-2.0-flash')

    # Construct the messages for the model
    messages = [
        {"role": "user", "parts": [question]}
    ]

    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "parts": [msg.content]})
        else:
            messages.append({"role": "model", "parts": [msg]})

    response = model.generate_content(messages)
    return response.text


def extract_mermaid_code(text):
    """Extracts MermaidJS code from a string, removing ```mermaid and ```."""
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        return mermaid_match.group(1).strip()
    else:
        return text.strip()  # Return the original text if no mermaid code is found


def render_mermaid_html(mermaid_code):
    """Renders MermaidJS code as HTML."""
    html_content = f"""
    <!doctype html>
    <html lang="en">
      <body>
        <pre class="mermaid">
          {mermaid_code}
        </pre>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true }});
        </script>
      </body>
    </html>
    """
    return html_content


# Set up Streamlit page
st.set_page_config(page_title="LangChain + MermaidJS", layout="wide")
st.title("dictate for Diagram ")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mermaid_code" not in st.session_state:
    st.session_state.mermaid_code = """graph TD\n    A[Start] --> B[User Input]"""

# Chat and LLM section
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Chat")
    user_input = st.text_input("Enter your input:", key="user_input")

    if user_input:
        # Append user message
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Construct system prompt
        system_prompt = (
            "You are a system that converts user requirements into MermaidJS diagram code. "
            "Update the existing diagram using the user' input, and return ONLY the full MermaidJS code. Do not include any other text in your response. "
            "Return the MermaidJS code within ```mermaid and ```"
        )

        # Get response from Gemini
        response = get_gemini_response(system_prompt, st.session_state.chat_history)

        # Extract and clean the mermaid code
        clean_mermaid_code = extract_mermaid_code(response)

        # Append LLM response (just the string)
        st.session_state.chat_history.append(clean_mermaid_code)

        # Update MermaidJS code (directly assign the string)
        st.session_state.mermaid_code = clean_mermaid_code

    # Display chat history (only user messages)
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**User:** {msg.content}")

# Mermaid Diagram section
with col2:
    st.subheader(" Here is you updated")
    mermaid_html = render_mermaid_html(st.session_state.mermaid_code)
    st.components.v1.html(mermaid_html, height=500, scrolling=True)  # adjust height as needed