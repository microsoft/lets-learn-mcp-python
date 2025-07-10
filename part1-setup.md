# Part 1: Prerequisites and Setup (Python)

> **Navigation**: [← Back to Overview](README-python.md) | [Next: Part 2 - Game App →](part2-game-app-python.md)

## What You'll Need
Before diving into MCP development, ensure you have the following tools installed:

### 1. Visual Studio Code
- Download and install [VS Code](https://code.visualstudio.com/)
- Essential for MCP development and integration

### 2. Python 3.12+
- Install Python 3.12 or later from [Python.org](https://www.python.org/downloads/)
- Verify installation: `python --version` or `python3 --version`
- Ensure pip is installed: `pip --version` or `pip3 --version`

### 3. Python Extension for VS Code
- Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extension
- Provides comprehensive Python development support
- Includes IntelliSense, debugging, and virtual environment management

### 4. Node.js
- Download and install [Node.js](https://nodejs.org/) (LTS version recommended)
- Required for MCP Inspector and other MCP development tools
- Verify installation: `node --version` and `npm --version`
- Enables npm package management for MCP tooling

### 5. Docker Desktop
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Required for running MCP servers in containers
- Verify installation: `docker --version`
- Ensure Docker Desktop is running before starting MCP servers

### 6. Understanding MCP
- What is Model Context Protocol?
- How MCP Servers work with AI assistants
- The client-server architecture
- Use cases and benefits

## Environment Verification
- [ ] VS Code installed and running
- [ ] Python 3.12+ installed
- [ ] Python extension active in VS Code
- [ ] Node.js and npm installed
- [ ] Docker Desktop installed and running
- [ ] Basic understanding of MCP concepts

## What is Model Context Protocol (MCP)?

The Model Context Protocol (MCP) is an open standard that enables AI assistants to securely access and interact with external tools, data sources, and services. Think of it as a bridge that allows AI models like GitHub Copilot, ChatGPT, or Claude to:

- **Access Real-time Data**: Connect to databases, APIs, and live data sources
- **Execute Actions**: Perform operations like creating files, sending emails, or managing systems
- **Extend Capabilities**: Add custom functionality specific to your needs
- **Maintain Security**: Control what the AI can and cannot access

### Key Benefits for Python Developers

1. **Familiar Ecosystem**: Use your existing Python skills and rich library ecosystem
2. **Rich Tooling**: Leverage VS Code and Python extension for development
3. **Rapid Prototyping**: Python's simplicity enables quick MCP server development
4. **Async Support**: Built-in async/await support for efficient I/O operations
5. **Strong Typing**: Optional type hints with mypy for better code quality
6. **JSON Native**: Excellent JSON support with built-in libraries

### Architecture Overview

```
┌─────────────────┐    MCP Protocol    ┌──────────────────┐
│   AI Assistant  │ ◄─────────────────►│   MCP Server     │
│ (VS Code, etc.) │                    │  (Your Python    │
└─────────────────┘                    │     App)         │
                                       └──────────────────┘
                                              │
                                              ▼
                                       ┌──────────────────┐
                                       │  Your Services   │
                                       │ (Business Logic) │
                                       └──────────────────┘
```

### Use Cases

- **Data Analysis**: Connect AI to your databases and analytics systems
- **Automation**: Let AI assistants manage your development workflows
- **Documentation**: Generate and update documentation from live code
- **Testing**: Create and run tests based on AI analysis
- **Deployment**: Automate deployment processes with AI guidance
- **Web Scraping**: Build intelligent data collection tools
- **API Integration**: Create bridges to external services

## Python Virtual Environment Setup

It's highly recommended to use virtual environments for Python MCP development:

### Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv mcp-env

# Activate on macOS/Linux
source mcp-env/bin/activate

# Activate on Windows
mcp-env\Scripts\activate
```

### Install Core Dependencies
```bash
# Install MCP Python SDK
pip install mcp

# Install development dependencies
pip install asyncio aiofiles
```

### VS Code Python Environment
1. Open VS Code in your project folder
2. Press `Ctrl+Shift+P` (Cmd+Shift+P on Mac)
3. Type "Python: Select Interpreter"
4. Choose the interpreter from your virtual environment

---

> **Next Step**: Now that you understand MCP and have your environment set up, let's build your first application that uses existing MCP servers.

**Continue to**: [Part 2 - Building MyGame App →](part2-game-app-python.md)
