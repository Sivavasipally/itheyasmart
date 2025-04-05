# MCP Server with Gemini AI Integration

This project implements a Multi-Component Platform (MCP) server with Gemini AI integration, allowing users to perform various mathematical operations and complex tasks through natural language commands.

## Features

- **Mathematical Operations**
  - Basic arithmetic (add, subtract, multiply, divide)
  - Advanced math (power, square root, cube root)
  - Special functions (factorial, log, trigonometric functions)
  - List operations (sum of list, exponential sum)

- **String Processing**
  - Convert strings to ASCII values
  - Process character arrays

- **Keynote Integration**
  - Open Keynote application
  - Draw rectangles with custom dimensions
  - Add text to shapes

- **AI-Powered Task Execution**
  - Natural language processing using Gemini AI
  - Iterative problem solving
  - Automatic tool selection based on user queries

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- macOS (for Keynote integration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Project Structure

- `mcp_server.py`: Contains the MCP server implementation and tool definitions
- `talk2mcp.py`: Client application that interfaces with the MCP server and Gemini AI
- `.env`: Configuration file for API keys
- `requirements.txt`: Project dependencies

## Usage

You can start the application in two ways:

### Option 1: Start Server and Client Separately
1. Start the MCP server in one terminal:
```bash
python mcp_server.py
```

2. In another terminal, run the client application:
```bash
python talk2mcp.py
```

### Option 2: Start Client Only (Recommended)
The client application can automatically start the server if it's not already running. Simply run:
```bash
python talk2mcp.py
```

The client will:
1. Check if the server is running
2. Start the server if needed
3. Establish connection automatically
4. Prompt for your query

### Using the Application
1. Enter your query when prompted. Examples:
   - "Add 5 and 3"
   - "Find the ASCII values of characters in INDIA"
   - "Start keynote app and draw a rectangle of size 300x400"
   - "Calculate the factorial of 5"

2. Type 'exit' to quit the application.

Note: When using Option 2, the server will automatically shut down when you exit the client application.

## Available Tools

The system provides the following tools:

1. **Mathematical Tools**
   - `add(a: int, b: int)`: Add two numbers
   - `subtract(a: int, b: int)`: Subtract two numbers
   - `multiply(a: int, b: int)`: Multiply two numbers
   - `divide(a: int, b: int)`: Divide two numbers
   - `power(a: int, b: int)`: Calculate power
   - `sqrt(a: int)`: Calculate square root
   - `cbrt(a: int)`: Calculate cube root
   - `factorial(a: int)`: Calculate factorial
   - `log(a: int)`: Calculate logarithm
   - `sin(a: int)`, `cos(a: int)`, `tan(a: int)`: Trigonometric functions

2. **String Processing Tools**
   - `strings_to_chars_to_int(string: str)`: Convert string to ASCII values
   - `int_list_to_exponential_sum(int_list: list)`: Calculate sum of exponentials

3. **Keynote Tools**
   - `open_keynote()`: Open Keynote application
   - `draw_rectangle_in_keynote(shapeWidth: int, shapeHeight: int)`: Draw rectangle
   - `add_text_to_keynote_shape(text: str)`: Add text to shape
  
## Demo

Watch a demo of the MCP Server with Gemini AI integration in action:

[![MCP Server Demo](https://img.youtube.com/vi/N36YqaE25wA/0.jpg)](https://youtu.be/N36YqaE25wA?si=MQhxSDJhyokvaGUf)

Click the image above to watch the demo video on YouTube. 

## Error Handling

The system includes comprehensive error handling:
- Timeout handling for AI responses
- Type conversion validation
- Tool availability checking
- Parameter validation

## Debugging

Debug information is printed to the console, including:
- Tool execution details
- Parameter processing
- Result formatting
- Error messages and stack traces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for natural language processing capabilities
- MCP framework for tool management
- Python community for various libraries used in this project 