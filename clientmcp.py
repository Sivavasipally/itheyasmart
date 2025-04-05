import os
import time
import platform
import streamlit as st
from dotenv import load_dotenv
import asyncio
from google import genai
from concurrent.futures import TimeoutError
import traceback
import subprocess
import sys
import json

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="MCP Tool Interface",
    page_icon="🤖",
    layout="wide"
)

# Session state initialization - Make sure this happens BEFORE any functions attempt to use these values
if 'tools' not in st.session_state:
    st.session_state.tools = []
if 'iteration_responses' not in st.session_state:
    st.session_state.iteration_responses = []
if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = ""
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'available_models' not in st.session_state:
    st.session_state.available_models = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro"]
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash"
if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 10
if 'timeout' not in st.session_state:
    st.session_state.timeout = 10
if 'connection_method' not in st.session_state:
    st.session_state.connection_method = "tcp"  # Default to TCP connection
if 'server_process' not in st.session_state:
    st.session_state.server_process = None
if 'server_port' not in st.session_state:
    st.session_state.server_port = 8765
if 'mcp_server_path' not in st.session_state:
    st.session_state.mcp_server_path = "mcp_server.py"
if 'manual_tool_input' not in st.session_state:
    st.session_state.manual_tool_input = False


# Initialize Gemini client
@st.cache_resource
def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables")
        return None
    return genai.Client(api_key=api_key)


# Function to generate content with timeout
async def generate_with_timeout(client, prompt, timeout, model_name):
    """Generate content with a timeout"""
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        return response
    except TimeoutError:
        st.error(f"LLM generation timed out after {timeout} seconds!")
        raise
    except Exception as e:
        st.error(f"Error in LLM generation: {e}")
        raise


# Helper function to start MCP server with TCP
def start_tcp_server():
    try:
        cmd = [sys.executable, st.session_state.mcp_server_path, "--port", str(st.session_state.server_port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        st.session_state.server_process = process
        # Wait a moment for server to start
        time.sleep(2)
        return True
    except Exception as e:
        st.error(f"Failed to start MCP server: {e}")
        return False


# Function to stop the MCP server process
def stop_tcp_server():
    if st.session_state.server_process:
        try:
            st.session_state.server_process.terminate()
            st.session_state.server_process = None
            return True
        except Exception as e:
            st.error(f"Error stopping server: {e}")
            return False
    return True


# Function to manually add tool definitions
def add_manual_tools(tools_json):
    try:
        tools_data = json.loads(tools_json)

        # Simple Tool class to mimic MCP tool structure
        class Tool:
            def __init__(self, name, description, input_schema):
                self.name = name
                self.description = description
                self.inputSchema = input_schema

        # Convert JSON data to Tool objects
        tools = []
        for tool_data in tools_data:
            tool = Tool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {})
            )
            tools.append(tool)

        st.session_state.tools = tools
        return True
    except Exception as e:
        st.error(f"Error parsing tools JSON: {e}")
        return False


# Function to create tool descriptions for system prompt
def create_tools_description():
    tools_description = []
    for i, tool in enumerate(st.session_state.tools):
        try:
            params = tool.inputSchema
            desc = getattr(tool, 'description', 'No description available')
            name = getattr(tool, 'name', f'tool_{i}')

            if 'properties' in params:
                param_details = []
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'unknown')
                    param_details.append(f"{param_name}: {param_type}")
                params_str = ', '.join(param_details)
            else:
                params_str = 'no parameters'

            tool_desc = f"{i + 1}. {name}({params_str}) - {desc}"
            tools_description.append(tool_desc)
        except Exception as e:
            tools_description.append(f"{i + 1}. Error processing tool: {str(e)}")

    return "\n".join(tools_description)


# Function to create system prompt
def create_system_prompt():
    tools_description = create_tools_description()

    system_prompt = f"""You are an intelligent AI agent capable of solving math problems, answering user queries and completing 
    complex tasks asked by a user in iterations. You have access to various mathematical and complex tasks related tools.

Available tools:
{tools_description}

You must respond with EXACTLY ONE line in one of these formats (no additional text):
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...

2. For final answers:
   FINAL_ANSWER: [number]

Important:
- When a function returns multiple values, you need to process all of them
- Only give FINAL_ANSWER when you have completed all necessary calculations
- Do not repeat function calls with the same parameters

Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FINAL_ANSWER: [42]

DO NOT include any explanations or additional text.
Your entire response should be a single line starting with either FUNCTION_CALL: or FINAL_ANSWER:"""

    st.session_state.system_prompt = system_prompt
    return system_prompt


# Simulated tool call for demo mode or when actual MCP connection isn't available
async def simulate_tool_call(func_name, arguments):
    """Simulate a tool call with some basic functionality"""
    if func_name == "add":
        return sum(arguments.values())
    elif func_name == "multiply":
        result = 1
        for val in arguments.values():
            result *= val
        return result
    elif func_name == "strings_to_chars_to_int":
        input_str = list(arguments.values())[0]
        return [ord(c) for c in input_str]
    else:
        return f"Simulated response for {func_name} with arguments {arguments}"


# Process a query
async def process_query(query):
    try:
        st.session_state.is_processing = True
        st.session_state.iteration = 0
        st.session_state.iteration_responses = []
        st.session_state.last_response = None

        client = get_genai_client()
        if not client:
            st.error("Failed to initialize Gemini client")
            st.session_state.is_processing = False
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Get the current model from session state
        model_name = st.session_state.selected_model

        # Initialize the execution container for displaying iteration results
        execution_container = st.container()

        current_query = query

        while st.session_state.iteration < st.session_state.max_iterations:
            iteration_num = st.session_state.iteration + 1
            progress = min(1.0, st.session_state.iteration / st.session_state.max_iterations)
            progress_bar.progress(progress)
            status_text.text(f"Processing iteration {iteration_num}/{st.session_state.max_iterations}")

            with execution_container.expander(f"Iteration {iteration_num}", expanded=True):
                st.write(f"Query for this iteration:\n{current_query}")

                if st.session_state.last_response is not None:
                    current_query = current_query + "\n\n" + " ".join(st.session_state.iteration_responses)
                    current_query = current_query + "  What should I do next?"

                # Get model's response with timeout
                prompt = f"{st.session_state.system_prompt}\n\nQuery: {current_query}"

                try:
                    with st.spinner("Generating LLM response..."):
                        # Pass the model name directly instead of accessing session state
                        response = await generate_with_timeout(client, prompt, st.session_state.timeout, model_name)
                        response_text = response.text.strip()
                        st.write(f"LLM Response: {response_text}")

                    # Find the FUNCTION_CALL line in the response
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line.startswith("FUNCTION_CALL:") or line.startswith("FINAL_ANSWER:"):
                            response_text = line
                            break

                except Exception as e:
                    st.error(f"Failed to get LLM response: {e}")
                    break

                if response_text.startswith("FUNCTION_CALL:"):
                    _, function_info = response_text.split(":", 1)
                    parts = [p.strip() for p in function_info.split("|")]
                    func_name, params = parts[0], parts[1:]

                    try:
                        # Find the matching tool
                        tool = next((t for t in st.session_state.tools if t.name == func_name), None)
                        if not tool:
                            st.error(f"Unknown tool: {func_name}")
                            break

                        # Prepare arguments
                        arguments = {}
                        schema_properties = tool.inputSchema.get('properties', {})

                        for param_name, param_info in schema_properties.items():
                            if not params:
                                st.error(f"Not enough parameters provided for {func_name}")
                                break

                            value = params.pop(0)
                            param_type = param_info.get('type', 'string')

                            # Convert value to correct type
                            if param_type == 'integer':
                                arguments[param_name] = int(value)
                            elif param_type == 'number':
                                arguments[param_name] = float(value)
                            elif param_type == 'array':
                                if isinstance(value, str):
                                    value = value.strip('[]').split(',')
                                arguments[param_name] = [int(x.strip()) for x in value]
                            else:
                                arguments[param_name] = str(value)

                        st.write(f"Calling tool: {func_name}")
                        st.write(f"With arguments: {arguments}")

                        # Call the tool (here we'll use our simulation since we can't connect to actual MCP)
                        with st.spinner("Executing function..."):
                            # In a real implementation, we would call the MCP server here
                            # For now, we'll simulate it
                            iteration_result = await simulate_tool_call(func_name, arguments)

                        # Format result for display
                        if isinstance(iteration_result, list):
                            result_str = f"[{', '.join(map(str, iteration_result))}]"
                        else:
                            result_str = str(iteration_result)

                        st.success(f"Result: {result_str}")

                        # Update session state
                        response_text = f"In the {iteration_num} iteration you called {func_name} with {arguments} parameters, and the function returned {result_str}."
                        st.session_state.iteration_responses.append(response_text)
                        st.session_state.last_response = iteration_result

                    except Exception as e:
                        st.error(f"Error executing function: {str(e)}")
                        traceback.print_exc()
                        break

                elif response_text.startswith("FINAL_ANSWER:"):
                    final_answer = response_text.split(":", 1)[1].strip()
                    st.success(f"Final Answer: {final_answer}")
                    break

                st.session_state.iteration += 1

            # Small delay for better UI experience
            time.sleep(0.5)

        progress_bar.progress(1.0)
        status_text.text("Execution complete!")
        st.session_state.is_processing = False

    except Exception as e:
        st.error(f"Error processing query: {e}")
        traceback.print_exc()
        st.session_state.is_processing = False


# Sample tool definitions for manual setup
SAMPLE_TOOLS_JSON = '''[
  {
    "name": "add",
    "description": "Add two numbers together",
    "inputSchema": {
      "type": "object",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"}
      },
      "required": ["x", "y"]
    }
  },
  {
    "name": "multiply",
    "description": "Multiply two numbers together",
    "inputSchema": {
      "type": "object",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"}
      },
      "required": ["x", "y"]
    }
  },
  {
    "name": "strings_to_chars_to_int",
    "description": "Convert a string to a list of ASCII values",
    "inputSchema": {
      "type": "object",
      "properties": {
        "input": {"type": "string"}
      },
      "required": ["input"]
    }
  }
]'''

# Sidebar for settings
with st.sidebar:
    st.title("MCP Agent Settings")

    with st.expander("LLM Settings", expanded=True):
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            st.session_state.available_models
        )

        st.session_state.max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=30,
            value=st.session_state.max_iterations
        )

        st.session_state.timeout = st.slider(
            "LLM Timeout (seconds)",
            min_value=5,
            max_value=60,
            value=st.session_state.timeout
        )

    with st.expander("Server Connection", expanded=True):
        connection_options = ["Demo Mode", "TCP Connection", "Manual Tool Definition"]
        connection_choice = st.radio(
            "Connection Method",
            connection_options,
            index=0
        )

        if connection_choice == "TCP Connection":
            st.session_state.connection_method = "tcp"
            st.session_state.manual_tool_input = False

            st.session_state.server_port = st.number_input(
                "Server Port",
                min_value=1000,
                max_value=65535,
                value=st.session_state.server_port
            )

            st.session_state.mcp_server_path = st.text_input(
                "MCP Server Script Path",
                value=st.session_state.mcp_server_path
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Server",
                             disabled=st.session_state.is_processing or st.session_state.server_process is not None):
                    with st.spinner("Starting MCP server..."):
                        if start_tcp_server():
                            st.success("Server started!")
                        else:
                            st.error("Failed to start server")

            with col2:
                if st.button("Stop Server", disabled=st.session_state.server_process is None):
                    with st.spinner("Stopping server..."):
                        if stop_tcp_server():
                            st.success("Server stopped!")
                        else:
                            st.error("Failed to stop server")

        elif connection_choice == "Demo Mode":
            st.session_state.connection_method = "demo"
            st.session_state.manual_tool_input = False

            st.info("Demo mode uses simulated tool calls and predefined tools.")

            if st.button("Load Demo Tools"):
                with st.spinner("Loading demo tools..."):
                    if add_manual_tools(SAMPLE_TOOLS_JSON):
                        st.success(f"Loaded {len(st.session_state.tools)} demo tools!")
                        create_system_prompt()
                    else:
                        st.error("Failed to load demo tools")

        elif connection_choice == "Manual Tool Definition":
            st.session_state.connection_method = "manual"
            st.session_state.manual_tool_input = True

            st.info("Define tools manually in JSON format.")

            tools_json = st.text_area(
                "Tool Definitions (JSON)",
                value=SAMPLE_TOOLS_JSON,
                height=300
            )

            if st.button("Apply Tool Definitions"):
                with st.spinner("Processing tool definitions..."):
                    if add_manual_tools(tools_json):
                        st.success(f"Loaded {len(st.session_state.tools)} tools!")
                        create_system_prompt()
                    else:
                        st.error("Failed to load tool definitions")

    if st.session_state.tools:
        with st.expander("Available Tools", expanded=False):
            for i, tool in enumerate(st.session_state.tools):
                st.markdown(f"**{i + 1}. {tool.name}**")
                st.markdown(f"*{getattr(tool, 'description', 'No description')}*")
                st.markdown("---")

# Main content area
st.title("🤖 MCP Tool Interface")

st.markdown("""
This interface allows you to interact with the MCP (Machine Comprehension Protocol) server and execute complex tasks using AI.

**To get started:**
1. Select a connection method in the sidebar
2. Initialize tools using your preferred method
3. Enter your query or task below
4. Click "Execute Query" to run the task
""")

# Query input
query = st.text_area(
    "Enter your query or task:",
    height=100,
    placeholder="Example: Find the ASCII values of characters in INDIA and then take the sum of those values."
)

col1, col2 = st.columns([1, 6])
with col1:
    execute_button = st.button(
        "Execute Query",
        type="primary",
        disabled=st.session_state.is_processing or not st.session_state.tools
    )

with col2:
    if st.session_state.is_processing:
        st.info("Processing query... Please wait.")
    elif not st.session_state.tools:
        st.warning("Please initialize tools first using one of the methods in the sidebar.")

# Execute query when button is clicked
if execute_button and query and st.session_state.tools:
    asyncio.run(process_query(query))

# Display past iterations if any
if st.session_state.iteration_responses and not st.session_state.is_processing:
    st.header("Execution Summary")

    for i, response in enumerate(st.session_state.iteration_responses):
        with st.expander(f"Iteration {i + 1}", expanded=False):
            st.write(response)

    st.success("Task execution complete!")

# Display footer
st.markdown("---")
st.caption("MCP Tool Interface © 2025")