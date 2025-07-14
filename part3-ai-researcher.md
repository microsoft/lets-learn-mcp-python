# Part 3: Building Your Own MCP Server - AI Research Learning Hub

> **Navigation**: [← Part 2: Python Study Buddy](part2-study-buddy.md) | [Back to Overview](README.md)

## Overview

In this part, you'll build an AI Research Learning Hub that showcases advanced MCP server capabilities.

## Key Learning Objectives

1. **External MCP Integration** - Use Hugging Face and GitHub MCP servers to find research papers and code
2. **Local Data Management** - Create and query CSV and JSON storage systems
3. **Resource Exposure** - Make data accessible to AI via MCP resources
4. **GitHub Integration** - Find code implementations for research papers using GitHub MCP
5. **Intelligent Search** - Build flexible search across all research content

## Project Setup and Architecture

### Project Structure

Your completed project will have this structure:

```text
AIResearchHub/
├── server.py           # Main MCP server with all tools
├── paper_manager.py    # Handles paper discovery and CSV database
└── data/
    └── research_papers.csv  # Local paper database
```

### Essential Dependencies

Create your Python environment and install dependencies:

1. `uv venv` - Create a virtual environment
2. `source research-env/bin/activate` - Activate the environment (macOS/Linux) or `research-env\Scripts\activate` (Windows)
3. `uv sync` - Install required packages from `pyproject.toml`

### Configure MCP Integration

Create `.vscode/settings.json` to enable the MCP servers you'll need:

```json
{
    "servers": {
        "hf-mcp-server": {
            "url": "https://huggingface.co/mcp",
            "headers": {
                "Authorization": "Bearer YOUR_HUGGING_FACE_BEARER_TOKEN"
            }
        },
        "github": {
            "type": "http", 
            "url": "https://api.githubcopilot.com/mcp/"
        },
        "our-research-hub": {
            "command": "uv",
            "args": ["run", "python", "-m", "server"],
            "cwd": "./AIResearchHub",
            "env": {}
        }
    }
}
```

## Building the Paper Management System

### Paper Manager (`paper_manager.py`)

The paper manager handles discovery from Hugging Face and maintains a local CSV database. Key implementation highlights:

#### Core Architecture

```python
class PaperManager:
    """Manages AI/ML research papers using external MCP servers."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.csv_file = self.data_dir / "research_papers.csv"
        self._initialize_csv()
```

#### Smart Duplicate Prevention

The `add_papers_from_huggingface()` method includes robust duplicate detection:

- Checks existing paper IDs before adding new papers
- Validates required fields (ID and title)
- Handles different author and keyword data formats
- Returns count of actually added papers

#### Enhanced Search Capabilities

The search functionality covers multiple fields:

```python
# Search across title, abstract, keywords, and authors
mask = (
    df['title'].str.lower().str.contains(query_lower, case=False, na=False) |
    df['abstract'].str.lower().str.contains(query_lower, case=False, na=False) |
    df['keywords'].str.lower().str.contains(query_lower, case=False, na=False) |
    df['authors'].str.lower().str.contains(query_lower, case=False, na=False)
)
```

> **📁 Full Implementation**: See `AIResearchHub/paper_manager.py` for the complete code with all methods and error handling.

### Key Features Explained

- **CSV Database**: Stores papers with metadata including title, authors, abstract, keywords, and engagement metrics
- **Duplicate Prevention**: Checks existing paper IDs to avoid adding duplicates
- **Data Validation**: Handles missing data gracefully with proper type checking
- **Search Functionality**: Searches across title, abstract, keywords, and authors using pandas string operations
- **Statistics**: Provides insights into your paper collection
- **Paper Management**: Methods to retrieve, check existence, and remove papers

## MCP Server Implementation

### Main Server (`server.py`)

This is the heart of your AI Research Hub - a focused MCP server with essential research tools:

#### Server Architecture

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from paper_manager import PaperManager

mcp = FastMCP("AI Research Learning Hub")
paper_manager = PaperManager()
```

#### Key Tools Implementation

**1. Save Papers Tool** - Handles Hugging Face integration with robust error handling:

- Validates input data
- Reports papers saved vs skipped (duplicates)
- Returns comprehensive database statistics

**2. Search Papers Tool** - Provides flexible local search:

- Searches across title, abstract, keywords, and authors
- Returns limited results for display (5 papers) but reports total count
- Includes database statistics in response

**3. GitHub Integration Tool** - Generates optimized search strategies:

- Constructs targeted repository and code search queries
- Provides step-by-step guidance for using GitHub MCP tools
- Includes language-specific implementation hints

#### Resources and Prompts

```python
@mcp.resource("research://database/summary")
def get_database_summary() -> str:
    """Get a summary of the current state of the research database."""

@mcp.prompt(name="research_sprint")
def research_sprint_prompt(topic: str) -> str:
    """Guided workflow for comprehensive research on any topic."""
```

> **� Full Implementation**: See `AIResearchHub/server.py` for the complete code with all tool implementations, error handling, and response formatting.

### Complete Tool Set

Your AI Research Hub includes these core tools:

- **save_papers_to_database** - Save Hugging Face papers to local CSV with duplicate detection
- **search_local_papers** - Search your local paper database across multiple fields
- **find_code_implementations_of_papers** - Get optimized GitHub search strategies for finding paper implementations

### Resources Available

- **research://database/summary** - Complete database summary with paper statistics

### Prompts Available

- **research_sprint** - Guided workflow for comprehensive research on any topic

## 3.5 Testing and Using Your AI Research Hub

### Running Your MCP Server

## Testing and Using Your AI Research Hub

### Getting Started

1. **Start the Server**:

   ```bash
   cd AIResearchHub
   uv run python -m server
   ```

2. **Connect via VS Code**: Restart VS Code to load the new MCP server

3. **Test with AI Assistant**: Use GitHub Copilot to interact with your research hub

### Real-World Usage Examples

**Discover and Save Papers**:
> "Use Hugging Face to search for recent papers on transformer attention mechanisms, then use save_papers_to_database to add them to our local collection"

**Search Your Collection**:
> "Search our local papers database for anything related to diffusion models"

**Find GitHub Implementations**:
> "Use find_code_implementations_of_papers to get optimized search strategies for finding implementations of the Attention is All You Need paper"

**Get Database Overview**:
> "What's the current status of our research paper database?"

**Research Sprint**:
> "Perform a research sprint on transformer architectures" (uses the built-in prompt)

### Practical Workflows

#### Research Discovery Workflow

1. Use Hugging Face MCP to find papers on a topic
2. Use `save_papers_to_database` to add interesting papers to your local collection (automatically skips duplicates)
3. Use `find_code_implementations_of_papers` to get optimized GitHub search strategies
4. Execute the suggested GitHub searches using the GitHub MCP server
5. Organize and track your findings

#### Daily Research Routine

1. Check database stats with `search_local_papers` to see your research progress
2. Add new papers as you discover them using the database tools
3. Use the research sprint prompt for comprehensive topic exploration
4. Use GitHub integration to find practical implementations

#### GitHub Research Integration

1. Use `find_code_implementations_of_papers` to get optimized search queries
2. Execute the recommended GitHub MCP tools with the provided queries
3. Evaluate repositories based on the implementation hints provided
4. Build a curated list of high-quality research implementations

## Extending Your Research Hub

Your research hub is designed to be extensible. Here are some ideas for enhancements:

### Advanced Features to Consider

- **Study Notes System**: Add back note-taking capabilities with paper linking
- **Semantic Search**: Implement embedding-based search for better relevance
- **Citation Network Analysis**: Track paper relationships and citations
- **Automated Summaries**: Generate research summaries and insights
- **Collaboration Features**: Share collections and notes with team members

---

## Conclusion

**Congratulations!** You've built a robust AI Research Learning Hub that demonstrates core MCP server capabilities. Your system can:

- **Discover Research**: Connect with external MCP servers to find relevant papers
- **Manage Knowledge**: Store and organize papers locally with duplicate prevention
- **Enable Intelligence**: Provide AI assistants with rich research context via resources and prompts
- **Facilitate GitHub Integration**: Generate optimized search strategies for finding research implementations
- **Scale Gracefully**: Handle large collections of papers with efficient search and statistics

This project shows how MCP servers can transform how AI assistants help with research workflows. You've created a solid foundation that can be extended with additional features as your research needs grow.

**Next Steps**: Consider adding study notes management, semantic search capabilities, or automated literature review features. The MCP architecture you've built makes these enhancements straightforward to implement.

> **Continue Learning**: Explore the other parts of this tutorial series to see how MCP servers can enhance different workflows, from simple automation to complex domain expertise.
