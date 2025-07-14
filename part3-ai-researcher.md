# Part 3: Building Your Own MCP Server - AI Research Learning Hub

> **Navigation**: [← Part 2: Python Study Buddy](part2-study-buddy.md) | [Back to Overview](README.md)

## Overview

In this part, you'll build an AI Research Learning Hub that demonstrates core MCP server capabilities through a clean, minimal implementation. This project showcases how to create tools that work together, manage simple data storage, and prepare for integration with external MCP servers.

## Key Learning Objectives

1. **MCP Server Architecture** - Design tools that work together in a cohesive workflow
2. **Simple Data Management** - Use JSON storage for research tracking and organization
3. **External MCP Integration** - Prepare users for Hugging Face and GitHub MCP server usage
4. **Workflow Design** - Create guided research processes with multiple complementary tools
5. **Resource Exposure** - Make internal data accessible to AI assistants

## Project Setup and Architecture

### Project Structure

Your completed project will have this structure:

```text
AIResearchHub/
├── server.py                    # Main MCP server implementation
├── paper_manager.py             # JSON-based data management
└── data/
    └── research_papers.json    # Local research database
```

### Essential Dependencies

Create your Python environment and install dependencies:

1. `uv venv` - Create a virtual environment  
2. `source .venv/bin/activate` - Activate the environment (macOS/Linux) or `.venv\Scripts\activate` (Windows)
3. `uv sync` - Install required packages from `pyproject.toml`

> **Note**: The project uses `fastmcp>=2.10.5` and `mcp[cli]>=1.9.4` as core dependencies.

### Configure MCP Integration

Create `.vscode/mcp.json` to enable the MCP servers you'll need:

```json
{
    "inputs": [],
    "servers": {
        "huggingface": {
            "url": "https://huggingface.co/mcp",
            "headers": {
                "Authorization": "Bearer YOUR_HUGGING_FACE_BEARER_TOKEN"
            }
        },
        "github": {
            "type": "http", 
            "url": "https://api.githubcopilot.com/mcp/"
        },
        "ai-research-hub": {
            "command": "uv",
            "args": ["run", "python", "-m", "server"],
            "cwd": "./AIResearchHub",
            "env": {}
        }
    }
}
```

> **Important**: Replace `YOUR_HUGGING_FACE_BEARER_TOKEN` with your actual Hugging Face token for full functionality.

## Building the Paper Management System

### Paper Manager (`paper_manager.py`)

The paper manager provides a clean interface for managing research data using JSON storage:

#### Core Architecture

```python
class PaperManager:
    """Paper manager that handles research tracking and data storage."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.papers_file = self.data_dir / "research_papers.json"
    
    def add_research_entry(self, topic: str) -> dict[str, Any]:
        """Add a new research entry for a topic."""
        research_entry = {
            "id": len(papers) + 1,
            "topic": topic,
            "created": datetime.now().isoformat(),
            "status": "pending",
            "papers_found": [],
            "repositories_found": [],
            "notes": ""
        }
        return research_entry
```

#### Key Features

- **JSON Storage**: Human-readable data format that's easy to inspect and modify
- **Research Entries**: Structured tracking of research topics with unique IDs
- **Paper and Repository Management**: Add papers and repositories to research entries with timestamps
- **Status Tracking**: Monitor research progress from pending to active to complete
- **Search Functionality**: Search across research entries, papers, and repositories
- **Complete CRUD Operations**: Full create, read, update functionality for research management

> **📁 Full Implementation**: See `AIResearchHub/paper_manager.py` for the complete implementation with all methods.

## MCP Server Implementation

### Main Server (`server.py`)

The server demonstrates how to create complementary tools that work together in a research workflow:

#### Server Architecture

```python
from fastmcp import FastMCP
from paper_manager import PaperManager

mcp = FastMCP("AI Research Hub")
paper_manager = PaperManager()
```

#### Core Tools: Complete Research Management

**Tool 1: Research Entry Creation**

```python
@mcp.tool(description="Start researching a topic and get research ID")
async def research_topic(topic: str) -> dict:
    """Create research entry and provide research ID."""
```

This tool:

- Creates a structured research entry in the local database
- Returns a unique research ID for tracking
- Reports the total number of research topics
- Focuses solely on what it accomplished

**Tool 2: GitHub Search Strategy**

```python
@mcp.tool(description="Get GitHub search strategies for finding implementations")
async def get_github_searches(topic: str) -> dict:
    """Generate GitHub search strategies for finding code implementations."""
```

This tool:

- Generates multiple targeted search variations
- Provides specific GitHub MCP commands with quality filters
- Includes star count filters and language specifications
- Returns actionable search strategies

**Tool 3: Paper Management**

```python
@mcp.tool(description="Add a paper to a research entry")
async def add_paper(research_id: int, title: str, authors: str = "", url: str = "") -> dict:
    """Add a research paper to an existing research entry."""
```

**Tool 4: Repository Management**

```python
@mcp.tool(description="Add a repository to a research entry")
async def add_repository(research_id: int, name: str, url: str = "", stars: int = 0) -> dict:
    """Add a code repository to an existing research entry."""
```

**Tool 5: Research Search**

```python
@mcp.tool(description="Search local research database")
async def search_research(query: str) -> dict:
    """Search your local research database for matching content."""
```

**Tool 6: Status Management**

```python
@mcp.tool(description="Update research status and add notes")
async def update_research_status(research_id: int, status: str, notes: str = "") -> dict:
    """Update the status of a research entry."""
```

#### Guided Research Workflow

**Research Prompt** - Orchestrates the complete workflow:

```python
@mcp.prompt(name="research_workflow")
def research_workflow_prompt(topic: str) -> str:
    """Complete research workflow for any topic."""
```

The workflow provides:

1. Step-by-step instructions using all tools
2. Integration points with external MCP servers
3. Clear guidance for organizing findings
4. Focus areas for analysis and summary

#### Resource: Research Dashboard

**Comprehensive Status Resource**:

```python
@mcp.resource("status://dashboard")
def research_status() -> str:
    """Current research status and saved topics."""
```

Provides:

- Complete research activity overview
- Active vs pending research counts
- Detailed entries with timestamps
- Database statistics and health information

> **📁 Full Implementation**: See `AIResearchHub/server.py` for the complete server with all tools, prompts, and resources.

### Complete Tool Set

Your AI Research Hub includes:

- **research_topic** - Create and track research entries with unique IDs
- **get_github_searches** - Generate optimized GitHub search strategies with quality filters
- **add_paper** - Add research papers to existing research entries
- **add_repository** - Add code repositories to existing research entries
- **search_research** - Search your local research database for matching content
- **update_research_status** - Update research status and add notes
- **research_workflow** prompt - Guided workflow using all tools and external MCP servers
- **status://dashboard** resource - Comprehensive research dashboard and statistics

## Running and Using Your Research Hub

### Starting the Server

1. **Launch the Server**:

   ```bash
   cd AIResearchHub
   uv run python -m server
   ```

2. **Connect via IDE**: Restart your development environment to load the new MCP server

3. **Verify Connection**: Confirm the AI Research Hub appears in your MCP server list

## Usage Examples and Workflows

### Basic Research Operations

**Create Research Entry**:
> "Use research_topic to start researching 'transformer attention mechanisms'"

**Get GitHub Search Strategies**:
> "Use get_github_searches to get search strategies for 'neural networks'"

**Add Papers and Repositories**:
> "Use add_paper to add this paper to research #1: 'Attention Is All You Need' by Vaswani et al."

> "Use add_repository to add this repo to research #1: pytorch/pytorch with 50000 stars"

**Search Your Database**:
> "Use search_research to find all my research on 'transformers'"

**Update Research Status**:
> "Use update_research_status to mark research #1 as complete with notes about findings"

**Check Research Status**:
> "What's in my research log? Show me status://dashboard"

**Follow Complete Workflow**:
> "Start a research workflow for 'diffusion models'" (uses the research_workflow prompt)

### Integration with External MCP Servers

The tools work together for seamless external server usage:

1. **Create Research Foundation**: Use `research_topic("your topic")` to establish tracking
2. **Get Targeted Strategies**: Use `get_github_searches("your topic")` for optimized search commands
3. **Execute External Searches**: Run HuggingFace and GitHub MCP with provided strategies
4. **Save Findings**: Use `add_paper()` and `add_repository()` to store discoveries
5. **Monitor Progress**: Check `status://dashboard` for your complete research history
6. **Update Status**: Use `update_research_status()` to mark completion and add notes

### Example Tool Responses

**research_topic("neural networks") returns:**

```json
{
  "success": true,
  "topic": "neural networks",
  "research_id": 1,
  "message": "Research entry #1 created for 'neural networks'",
  "total_research_topics": 1
}
```

**get_github_searches("neural networks") returns:**

```json
{
  "success": true,
  "topic": "neural networks",
  "github_searches": [
    "neural networks implementation",
    "neural networks python"
  ],
  "commands": [
    "Search repos: neural networks stars:>10"
  ],
  "instructions": "Use GitHub MCP with these search terms"
}
```

## Research Workflows

### Comprehensive Research Process

1. **Initialize Research**: Create entry with topic and get research ID
2. **Plan GitHub Search**: Generate targeted search strategies with quality filters
3. **Execute External Searches**: Use HuggingFace MCP for papers, GitHub MCP for code
4. **Track Progress**: Monitor all research activities through the status resource
5. **Analyze Findings**: Compare academic papers with practical implementations

### Daily Research Routine

1. **Review Active Research**: Check status resource for ongoing projects
2. **Start New Topics**: Create entries for emerging research interests
3. **Execute Searches**: Use external MCP servers with generated strategies
4. **Update Findings**: Add discovered papers and repositories to research entries
5. **Synthesize Insights**: Compare theoretical advances with practical implementations

## Key Design Principles

### Tool Architecture
- **Single Responsibility**: Each tool has one clear, focused purpose
- **Complementary Functions**: Tools work together to support complete workflows
- **Clean Separation**: Data management separated from server logic
- **Extensible Design**: Ready for enhancement without major restructuring

### Data Management
- **Simple Storage**: JSON format for quick development and easy inspection
- **Structured Tracking**: Organized research entries with consistent schemas
- **Status Management**: Clear progression from pending to active research
- **Future-Ready**: Methods in place for adding papers and repositories

### Integration Strategy
- **External Preparation**: Tools prepare users for external MCP server usage
- **Clear Guidance**: Prompts provide actionable workflows
- **Quality Filters**: Built-in criteria for finding high-quality implementations
- **Resource Exposure**: Internal data accessible to AI assistants

## Extending Your Research Hub

### Immediate Enhancements
- **Real MCP Integration**: Replace preparation with actual API calls to external servers
- **Enhanced Data Models**: Add more detailed paper and repository metadata
- **Search Functionality**: Implement local search across research entries
- **Export Capabilities**: Generate reports and summaries from research data

### Advanced Features
- **Semantic Search**: Embedding-based search for better content discovery
- **Citation Analysis**: Track relationships and influence between papers
- **Automated Summarization**: Generate insights from research collections
- **Collaboration Tools**: Share research with team members and track contributions
- **Data Visualization**: Create charts and graphs of research trends and progress

### Migration to Production

When scaling up:

1. **Enhanced Storage**: Move from JSON to databases for better performance and querying
2. **Real-Time Integration**: Direct API integration with HuggingFace and GitHub
3. **Advanced Tools**: Add specialized search, analysis, and reporting capabilities
4. **Error Handling**: Implement robust error handling and recovery mechanisms
5. **Authentication**: Add user management and access controls
6. **Monitoring**: Implement logging and metrics for system health

---

## Conclusion

**Congratulations!** You've built a functional AI Research Learning Hub that demonstrates essential MCP server concepts. Your system successfully:

- **Organizes Research**: Structures research activities with clear tracking and status management
- **Guides Workflows**: Provides step-by-step processes using complementary tools
- **Prepares Integration**: Generates optimized strategies for external MCP server usage
- **Exposes Data**: Makes research information accessible to AI assistants through resources
- **Scales Thoughtfully**: Provides a foundation that can grow with your research needs

## Key Achievements

Through this project, you've learned:

- **Tool Design Patterns**: How to create tools that work together without redundancy
- **Data Architecture**: Simple yet extensible approaches to data management
- **Workflow Orchestration**: Using prompts to guide complex multi-step processes
- **Resource Design**: Making internal data accessible and useful to AI assistants
- **Integration Planning**: Preparing for external service integration

**Next Steps**:

- Experiment with real external MCP integrations
- Add more sophisticated data analysis capabilities
- Explore advanced search and discovery features
- Consider collaboration and sharing functionality

> **Continue Learning**: This project demonstrates that powerful MCP servers start with clear, simple designs. The patterns you've learned here will serve as the foundation for much more sophisticated research and knowledge management systems.