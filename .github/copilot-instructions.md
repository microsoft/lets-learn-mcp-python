# LETSLEARNMCP-PYTHON Project Guide

## Project Overview
This is a tutorial series for building Model Context Protocol (MCP) servers in Python. The project includes:

1. **Python Study Buddy** - An interactive console application using MCP to help learn Python concepts
2. **AI Research Learning Hub** - An advanced MCP server for finding, tracking, and studying AI/ML research papers

## Architecture

### Overall Structure
- Tutorial-based repository with progressive complexity
- Python 3.12+ with modern features (async/await, type hints, dataclasses)
- MCP server implementation with tools and resources
- Data persistence using CSV and JSON storage

### Key Components
- **AIResearchHub/** - Main implementation of AI Research MCP server
  - `server.py` - Core MCP server with tools and resources
  - `paper_manager.py` - Manages research paper database (CSV)
  - `study_notes.py` - Handles study notes storage (JSON)
  - `data/` - Storage for papers and notes

## Data Flow
1. **Research Papers**: External sources (Hugging Face) → `paper_manager.py` → CSV database
2. **Study Notes**: User input → `study_notes.py` → JSON storage
3. **MCP Tools**: AI requests → server.py → appropriate manager → structured response
4. **Resources**: AI requests → resource endpoints → data access

## Environment Setup
```bash
# Create virtual environment
python -m venv research-env
source research-env/bin/activate  # macOS/Linux
# research-env\Scripts\activate   # Windows

# Install dependencies
pip install mcp pandas
```

## MCP Server Configuration
Configure in VS Code with `.vscode/settings.json`:
```json
{
  "mcp.servers": {
    "huggingface": {
      "command": "npx",
      "args": ["-y", "@huggingface/mcp-server"]
    },
    "github": {
      "command": "npx", 
      "args": ["-y", "@github/mcp-server"]
    },
    "our-research": {
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "./AIResearchHub"
    }
  }
}
```

## Development Workflow
1. **Run Server**: Navigate to AIResearchHub and execute `python -m server`
2. **Test MCP Tools**: Use VS Code with GitHub Copilot to interact with tools
3. **Verify Data**: Check CSV/JSON files in data/ directory after operations
4. **Debug Issues**: Review exception handling in tool responses

## Coding Standards
- **File Structure**: Separate manager classes for different data types
- **Type Hints**: Always use proper typing for parameters and returns
- **Error Handling**: Wrap tool implementations in try/except with structured error responses
- **JSON Responses**: All tools return consistent JSON structure with success/error status
- **Docstrings**: All classes and methods have descriptive docstrings
- **Resource Naming**: Use `research://` prefix for resource endpoints

## Key Design Patterns
1. **MCP Server Pattern**: Tools and resources exposed via FastMCP
2. **Repository Pattern**: Manager classes that handle data access
3. **Facade Pattern**: Server presenting simplified interface to complex subsystems
4. **Decorator Pattern**: MCP decorators for tools and resources
5. **Async Pattern**: Async tool implementations for concurrent operations

## Important Conventions
- All filenames use snake_case
- Classes use PascalCase
- Methods and variables use snake_case
- MCP tools and resources are explicitly annotated with descriptions
- JSON database files use indentation for readability
- All exceptions are caught and return structured error responses
- Paper IDs and note IDs are preserved across operations

## Example Tool Usage
```python
# Find papers using Hugging Face MCP
papers = await huggingface.search_papers("transformer attention")

# Save to local database
result = await our_research.save_papers_to_database(papers)

# Create study note
note = await our_research.add_study_note(
    title="Attention Mechanism Insights", 
    content="Key observations about transformer attention...",
    paper_id="2401.12345",
    tags=["attention", "transformers", "nlp"]
)

# Search local papers
results = await our_research.search_local_papers("attention")
```

## Resource Access
```python
# Access the papers database
papers_csv = await mcp.resource("research://papers/database")

# Access study notes
notes_json = await mcp.resource("research://notes/database")

# Access GitHub trending data
trending = await mcp.resource("research://github/trending")
```
