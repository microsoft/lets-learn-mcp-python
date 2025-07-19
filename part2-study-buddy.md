# Part 2: Using MCP Servers - Building Python Study Buddy App

## Overview
In this section, you'll learn how to use MCP servers by building an interactive Python learning companion that helps developers at different skill levels master Python concepts. This hands-on tutorial will teach you how MCP servers work from a consumer perspective before you build your own advanced server.

<img src="./images/study_buddy_app.png" alt="study-buddy-app" width="900" height="600">

## Step-by-Step Walkthrough

### 2.1 Project Setup

If you haven't already set up your python environment:

### 2.1.1. Visual Studio Code
- Download and install [VS Code](https://code.visualstudio.com/)
- Essential for MCP development and integration

### 2.1.2. Python 3.12+
- Install Python 3.12 or later from [Python.org](https://www.python.org/downloads/)
- Verify installation: `python --version` or `python3 --version`
- Ensure pip is installed: `pip --version` or `pip3 --version`

### 2.1.3. Python Extension for VS Code
- Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extension
- Provides comprehensive Python development support
- Includes IntelliSense, debugging, and virtual environment management

### 2.1.4. Install UV
To install UV, run the following command in the terminal:

```bash
pip install uv 
```

### 2.1.5. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv mcp-env

# Activate on macOS/Linux
source mcp-env/bin/activate

# Activate on Windows
mcp-env\Scripts\activate
```

### 2.1.6. Install packages 

```bash
uv sync 
```

### 2.2 MCP Server Configuration and Initial Setup

#### Configure Python Learning MCP
- Install and configure a Python Learning MCP server in VS Code
- to get the path to uv, run `which uv` 

Open the **.vscode** folder and make sure you see a file named **mcp.json** with the following content:

```json
"learnpython-mcp": {
         "command": "{path-to-uv}",
         "args": [
            "--directory",
            ".",
            "run",
            "server_part_2.py"
         ]
      }
```

### 2.2 Learning about Prompts 

- To get started we will learn how to use MCP prompts. To do this, edit the server in the **mcp.json** file to be `prompts_server.py` instead of `server_part_2.py`. 
- Next run the play button to start the server. 
- Examine the [prompts_server.py](./prompts_server.py) file and note the `@mcp.prompts` decorator. This is the prompt we will be looking for in copilot. 
- In Copilot chat type `\` and look for the prompt that you saw in the previous step. 
- Add the `beginner` variable to generate a beginner list of topics and press enter. 
- Once the prompt shows up in your terminal press enter and note the result! 

**Key Learning Points:**
- Prompts are great for storing information 
- Prompts can be dynamic and take variables 
- Prompts need to explicilty be called by the user unless sampling is done. 

### 2.3 Learning about Tools and Sampling

- To learn about MCP tools, edit the server in the **mcp.json** file to be `tools_server.py` instead of `prompts_server.py`. 
- Next run the play button to start the server. 
- Examine the [`tools_server.py`](./tools_server.py) file and note the `@mcp.tool()` decorator. These are the tools we will be using in Copilot.
- In Copilot chat, you can now use the tools to generate exercises. Try asking: "Use the generate_and_create_exercises tool to create 5 beginner Python exercises about variables"
- Notice how this tool uses **sampling** - it calls the LLM to generate the content and then processes the result
- You can also use the `list_exercises` tool to see all created exercises
- Try creating exercises for different levels (beginner, intermediate, advanced) and topics

**Key Learning Points:**
- Tools extend what LLMs can do by calling functions
- Sampling allows tools to use the LLM's capabilities within the tool execution
- Tools can work together to create more complex workflows

### 2.4 Learning about Resources 

- To learn about MCP resources, edit the server in the **mcp.json** file to be `resources_server.py` instead of `tools_server.py`. 
- Next run the play button to start the server. 
- Examine the [`resources_server.py`](./resources_server.py) file and note the `@mcp.resource()` decorator. These are file-like data sources that can be read by clients.
- Resources in this server provide access to:
  - `user://study-progress/{username}` - Get study progress for a specific user
  - `user://exercises/{level}` - List exercises available for a specific level
- You can also use the `get_users_progress` tool that combines resources with sampling to provide intelligent analysis
- Try asking Copilot to "Get the study progress for Marlene" and see how it accesses the resource
- Resources are different from tools - they provide data that can be read, while tools perform actions

**Key Learning Points:**
- Resources provide structured data access to LLMs
- Resources can be dynamically accessed with parameters (like username or level)
- Resources enable LLMs to read current state and context information

### 2.5 Running the Study Buddy App

Now let's use the full Study Buddy application with all MCP features integrated:

- Edit the server in the **mcp.json** file back to `server_part_2.py` (this is the complete server with all features) 
- Run the play button to start the server
- The [`server_part_2.py`](./server_part_2.py) includes:
  - **Prompts**: `python_topics` for generating topic lists
  - **Tools**: `generate_exercises`, `create_exercise`, `track_study_progress`, `start_study_buddy`
  - **Resources**: Access to exercises and progress data
- Open the **.Github** folder and read the **copilot-instructions.md** file. This will give copilot instructions on the workflow! This makes the experience more predictable. 

**To run the complete Study Buddy experience:**

1. **Start Study Session**: Tell Copilot in the chat that you want to learn Python and then follow the instructions. 

**What you'll experience:**
- 🐍 Welcome screen with your personalized progress
- 📚 Menu of available exercises at your level
- 💡 Hints and solutions for each exercise
- ✅ Progress tracking as you complete exercises
- 🎯 Achievement system to motivate continued learning

When the final command is run by copilot in the terminal you should observe the following in the terminal:

**Sample Study Session Output:**
```
🐍 Python Study Buddy

Welcome, CodeLearner! 
Ready to practice Python at the beginner level?

Current Streak: 🔥 0 days
Exercises Completed: ✅ 0

Available Beginner Exercises
┌────┬─────────────────┬────────────┬──────────────┐
│ No.│ Title           │ Difficulty │ Status       │
├────┼─────────────────┼────────────┼──────────────┤
│ 1  │ Hello World     │ ⭐         │ 📝 Not started │
│ 2  │ Variables       │ ⭐⭐       │ 📝 Not started │
└────┴─────────────────┴────────────┴──────────────┘

Choose an exercise (number): 1
```

The Study Buddy app demonstrates the full power of MCP by combining prompts, tools, resources, and sampling to create a personalized, interactive learning experience!


---

> **Next Step**: Now that you've built a learning application with MCP, let's create an advanced MCP server for AI research discovery!

**Continue to**: [Part 3 - Building Your AI Research Learning Hub →](part3-ai-researcher.md)
