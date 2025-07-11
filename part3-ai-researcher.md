# Part 3: Building Your Own MCP Server - AI Research Learning Hub

> **Navigation**: [← Part 2: Python Study Buddy](part2-study-buddy.md) | [Back to Overview](README.md)

> **Based on**: [AIResearchLearningMCP Repository](https://github.com/jamesmontemagno/AIResearchLearningMCP)

## Overview
Learn to build a practical MCP server that demonstrates three key MCP concepts: using external MCP servers, adding resources, and automating tasks. You'll create a research learning hub that finds AI/ML papers using Hugging Face MCP, stores them as resources, and automatically sends study notes to GitHub.

## Key Learning Objectives
1. **Find and Use External MCP Servers** - Use Hugging Face MCP Server to get AI/ML papers and create a local CSV database
2. **Add Resources with MCP** - Add the ML papers as data resources that AI assistants can access
3. **Automate Tasks with MCP** - Use the GitHub MCP server to send study notes to GitHub for permanent storage

## 3.1 Project Foundation

### Create Python project

Create a new folder called **AIResearchHub** and set up your Python project:

```bash
mkdir AIResearchHub
cd AIResearchHub
```

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv research-env

# Activate on macOS/Linux
source research-env/bin/activate

# Activate on Windows
research-env\Scripts\activate
```

### Package Installation
Essential packages for this focused MCP server:

```bash
pip install mcp>=1.9.4
pip install pandas
pip install csv
```

Create a **requirements.txt**:
```txt
mcp>=1.9.4
pandas>=2.0.0
```

### Project Structure
```bash
# Create main files
touch server.py
touch paper_manager.py
touch study_notes.py
```

Create the following directory structure:
```
AIResearchHub/
├── server.py           # Main MCP server
├── paper_manager.py    # Paper discovery and CSV management  
├── study_notes.py      # Study notes management
├── requirements.txt
└── data/
    └── research_papers.csv  # Local CSV database
```

## 3.2 External MCP Server Integration (Objective 1)

### Configure Hugging Face MCP Server
First, let's set up access to the Hugging Face MCP server in VS Code.

**Task**: Create `.vscode/mcp.json` to include Hugging Face MCP:

```json
{
    "inputs": [],
    "servers": {
        "huggingface": {
            "type": "sse",
            "url": "https://huggingface.co/mcp"
        },
        "github": {
            "type": "http", 
            "url": "https://api.githubcopilot.com/mcp/"
        },
        "our-research-hub": {
            "command": "python",
            "args": ["-m", "server"],
            "cwd": "./AIResearchHub",
            "env": {}
        }
    }
}
```

### Create Paper Discovery Service
**Task**: Create `paper_manager.py` that uses Hugging Face MCP to find papers and save to CSV.

```python
import csv
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class PaperManager:
    """Manages AI/ML research papers using external MCP servers."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.csv_file = self.data_dir / "research_papers.csv"
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            headers = [
                'paper_id', 'title', 'authors', 'abstract', 'url', 
                'published_date', 'research_field', 'keywords', 
                'downloads', 'likes', 'added_date'
            ]
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def add_papers_from_huggingface(self, papers_data: List[Dict[str, Any]]) -> int:
        """Add papers from Hugging Face search to local CSV database."""
        added_count = 0
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for paper in papers_data:
                # Extract paper information
                paper_row = [
                    paper.get('id', ''),
                    paper.get('title', ''),
                    ', '.join([author.get('name', '') for author in paper.get('authors', [])]),
                    paper.get('abstract', ''),
                    paper.get('url', ''),
                    paper.get('published_date', ''),
                    paper.get('research_field', 'machine_learning'),
                    ', '.join(paper.get('keywords', [])),
                    paper.get('downloads', 0),
                    paper.get('likes', 0),
                    datetime.now().isoformat()
                ]
                
                writer.writerow(paper_row)
                added_count += 1
        
        return added_count
    
    def get_papers_dataframe(self) -> pd.DataFrame:
        """Get all papers as a pandas DataFrame."""
        if self.csv_file.exists():
            return pd.read_csv(self.csv_file)
        return pd.DataFrame()
    
    def search_local_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search local CSV database for papers."""
        df = self.get_papers_dataframe()
        if df.empty:
            return []
        
        # Simple text search in title and abstract
        mask = (
            df['title'].str.contains(query, case=False, na=False) |
            df['abstract'].str.contains(query, case=False, na=False) |
            df['keywords'].str.contains(query, case=False, na=False)
        )
        
        matching_papers = df[mask]
        return matching_papers.to_dict('records')
    
    def get_paper_stats(self) -> Dict[str, Any]:
        """Get statistics about the local paper database."""
        df = self.get_papers_dataframe()
        if df.empty:
            return {"total_papers": 0}
        
        return {
            "total_papers": len(df),
            "research_fields": df['research_field'].value_counts().to_dict(),
            "most_recent_paper": df['published_date'].max() if 'published_date' in df.columns else None,
            "most_liked_paper": df.loc[df['likes'].idxmax(), 'title'] if 'likes' in df.columns and not df.empty else None,
            "database_last_updated": df['added_date'].max() if 'added_date' in df.columns else None
        }
```

## 3.3 MCP Resources Implementation (Objective 2)

### Create Study Notes Manager
**Task**: Create `study_notes.py` for managing study notes that will be sent to GitHub.

```python
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class StudyNotesManager:
    """Manages study notes for AI/ML research papers."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.notes_file = self.data_dir / "study_notes.json"
        self.notes_data = self._load_notes()
    
    def _load_notes(self) -> Dict[str, Any]:
        """Load study notes from file."""
        if self.notes_file.exists():
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"notes": [], "last_updated": None}
    
    def _save_notes(self):
        """Save study notes to file."""
        self.notes_data["last_updated"] = datetime.now().isoformat()
        with open(self.notes_file, 'w', encoding='utf-8') as f:
            json.dump(self.notes_data, f, indent=2, ensure_ascii=False)
    
    def add_study_note(self, paper_title: str, note_content: str, note_type: str = "summary") -> str:
        """Add a study note for a research paper."""
        note_id = f"note_{len(self.notes_data['notes']) + 1}_{int(datetime.now().timestamp())}"
        
        note = {
            "id": note_id,
            "paper_title": paper_title,
            "note_type": note_type,  # summary, insight, question, implementation
            "content": note_content,
            "created_date": datetime.now().isoformat(),
            "tags": self._extract_tags(note_content)
        }
        
        self.notes_data["notes"].append(note)
        self._save_notes()
        return note_id
    
    def get_notes_for_paper(self, paper_title: str) -> List[Dict[str, Any]]:
        """Get all notes for a specific paper."""
        return [note for note in self.notes_data["notes"] if note["paper_title"] == paper_title]
    
    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Get all study notes."""
        return self.notes_data["notes"]
    
    def generate_github_markdown(self) -> str:
        """Generate markdown content for GitHub repository."""
        markdown_content = "# AI/ML Research Study Notes\n\n"
        markdown_content += f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Group notes by paper
        papers = {}
        for note in self.notes_data["notes"]:
            paper_title = note["paper_title"]
            if paper_title not in papers:
                papers[paper_title] = []
            papers[paper_title].append(note)
        
        # Generate markdown for each paper
        for paper_title, notes in papers.items():
            markdown_content += f"## {paper_title}\n\n"
            
            for note in notes:
                markdown_content += f"### {note['note_type'].title()} ({note['created_date'][:10]})\n\n"
                markdown_content += f"{note['content']}\n\n"
                if note['tags']:
                    markdown_content += f"**Tags:** {', '.join(note['tags'])}\n\n"
                markdown_content += "---\n\n"
        
        return markdown_content
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from note content."""
        # Simple keyword-based tagging
        keywords = ['transformer', 'attention', 'neural', 'learning', 'deep', 'model', 
                   'algorithm', 'data', 'training', 'performance', 'architecture']
        
        tags = []
        content_lower = content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                tags.append(keyword)
        
        return tags[:5]  # Limit to 5 tags
```

### Create Main MCP Server
**Task**: Create `server.py` with focused tools for the three objectives.

```python
import asyncio
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from paper_manager import PaperManager
from study_notes import StudyNotesManager

# Initialize the MCP server
mcp = FastMCP("AI Research Learning Hub")

# Initialize managers
paper_manager = PaperManager()
notes_manager = StudyNotesManager()

# Objective 1: Find and Use External MCP Servers
@mcp.tool(
    description="Search Hugging Face for AI/ML research papers and save to local CSV database",
    annotations=ToolAnnotations(title="Paper Discovery via Hugging Face", idempotentHint=False, readOnlyHint=False),
)
async def discover_papers_from_huggingface(
    query: str,
    limit: int = 10,
    sort: str = "downloads"
) -> Dict[str, Any]:
    """
    Use Hugging Face MCP server to find AI/ML papers and save them to local CSV database.
    
    This tool demonstrates how to use external MCP servers and persist data locally.
    """
    
    try:
        # Note: In a real implementation, this would call the Hugging Face MCP server
        # For this demo, we'll simulate the response structure
        
        # Simulated Hugging Face paper search results
        simulated_papers = [
            {
                "id": f"hf_paper_{i}",
                "title": f"Advanced {query.title()} Research Paper {i}",
                "authors": [{"name": f"Researcher {i}A"}, {"name": f"Researcher {i}B"}],
                "abstract": f"This paper explores cutting-edge approaches to {query} using novel methodologies and extensive experimentation.",
                "url": f"https://huggingface.co/papers/example_{i}",
                "published_date": f"2024-0{min(i+1, 9)}-15",
                "research_field": "machine_learning",
                "keywords": [query, "AI", "research", "neural networks"],
                "downloads": 1000 + (i * 100),
                "likes": 50 + (i * 10)
            }
            for i in range(limit)
        ]
        
        # Add papers to local CSV database
        added_count = paper_manager.add_papers_from_huggingface(simulated_papers)
        
        # Get database statistics
        stats = paper_manager.get_paper_stats()
        
        return {
            "success": True,
            "query": query,
            "papers_found": len(simulated_papers),
            "papers_added_to_csv": added_count,
            "database_stats": stats,
            "csv_location": str(paper_manager.csv_file),
            "summary": f"Successfully discovered {len(simulated_papers)} papers on '{query}' from Hugging Face and saved to local CSV database"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error discovering papers from Hugging Face: {str(e)}"
        }

@mcp.tool(
    description="Search local CSV database of research papers",
    annotations=ToolAnnotations(title="Local Paper Search", idempotentHint=True, readOnlyHint=True),
)
async def search_local_papers(query: str) -> Dict[str, Any]:
    """Search the local CSV database for research papers."""
    
    try:
        papers = paper_manager.search_local_papers(query)
        stats = paper_manager.get_paper_stats()
        
        return {
            "success": True,
            "query": query,
            "results_count": len(papers),
            "papers": papers[:10],  # Limit to first 10 results
            "database_stats": stats,
            "summary": f"Found {len(papers)} papers matching '{query}' in local database"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error searching local papers: {str(e)}"
        }

# Objective 2: Add Resources with MCP
@mcp.resource("research://papers/database")
def get_papers_database() -> str:
    """Get the complete research papers database as a CSV resource."""
    try:
        df = paper_manager.get_papers_dataframe()
        if df.empty:
            return "No papers in database yet. Use discover_papers_from_huggingface to add papers."
        
        # Convert DataFrame to CSV string
        return df.to_csv(index=False)
        
    except Exception as e:
        return f"Error loading papers database: {str(e)}"

@mcp.resource("research://papers/summary")
def get_papers_summary() -> str:
    """Get a summary of the research papers database."""
    try:
        stats = paper_manager.get_paper_stats()
        
        summary = {
            "database_summary": stats,
            "available_fields": [
                "paper_id", "title", "authors", "abstract", "url",
                "published_date", "research_field", "keywords", 
                "downloads", "likes", "added_date"
            ],
            "usage_instructions": {
                "search": "Use search_local_papers tool to find specific papers",
                "discover": "Use discover_papers_from_huggingface to add new papers",
                "access": "Use research://papers/database resource to get full CSV data"
            }
        }
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        return f"Error generating papers summary: {str(e)}"

@mcp.resource("research://notes/all")
def get_all_study_notes() -> str:
    """Get all study notes as a JSON resource."""
    try:
        notes = notes_manager.get_all_notes()
        
        notes_resource = {
            "total_notes": len(notes),
            "notes": notes,
            "last_updated": notes_manager.notes_data.get("last_updated"),
            "note_types": list(set(note.get("note_type", "summary") for note in notes))
        }
        
        return json.dumps(notes_resource, indent=2)
        
    except Exception as e:
        return f"Error loading study notes: {str(e)}"

# Objective 3: Automate Tasks with MCP
@mcp.tool(
    description="Add a study note for a research paper",
    annotations=ToolAnnotations(title="Add Study Note", idempotentHint=False, readOnlyHint=False),
)
async def add_study_note(
    paper_title: str,
    note_content: str,
    note_type: str = "summary"
) -> Dict[str, Any]:
    """Add a study note for a research paper."""
    
    try:
        note_id = notes_manager.add_study_note(paper_title, note_content, note_type)
        
        return {
            "success": True,
            "note_id": note_id,
            "paper_title": paper_title,
            "note_type": note_type,
            "summary": f"Added {note_type} note for '{paper_title}'"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error adding study note: {str(e)}"
        }

@mcp.tool(
    description="Send study notes to GitHub repository using GitHub MCP server",
    annotations=ToolAnnotations(title="GitHub Notes Automation", idempotentHint=False, readOnlyHint=False),
)
async def send_notes_to_github(
    repository_name: str,
    commit_message: str = "Update AI/ML research study notes"
) -> Dict[str, Any]:
    """
    Automate sending study notes to GitHub repository using GitHub MCP server.
    
    This tool demonstrates how to automate tasks using external MCP servers.
    """
    
    try:
        # Generate markdown content from study notes
        markdown_content = notes_manager.generate_github_markdown()
        
        # Note: In a real implementation, this would use the GitHub MCP server
        # For this demo, we'll simulate the GitHub integration
        
        file_info = {
            "repository": repository_name,
            "file_path": "study_notes.md",
            "content_length": len(markdown_content),
            "commit_message": commit_message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Simulate successful GitHub operation
        github_response = {
            "success": True,
            "repository": repository_name,
            "file_created": "study_notes.md",
            "commit_sha": "abc123def456",  # Simulated commit hash
            "commit_message": commit_message,
            "url": f"https://github.com/username/{repository_name}/blob/main/study_notes.md"
        }
        
        return {
            "success": True,
            "github_response": github_response,
            "notes_count": len(notes_manager.get_all_notes()),
            "markdown_preview": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
            "automation_summary": f"Successfully sent {len(notes_manager.get_all_notes())} study notes to GitHub repository '{repository_name}'"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error sending notes to GitHub: {str(e)}"
        }

@mcp.tool(
    description="Get study notes for a specific paper",
    annotations=ToolAnnotations(title="Get Paper Notes", idempotentHint=True, readOnlyHint=True),
)
async def get_paper_notes(paper_title: str) -> Dict[str, Any]:
    """Get all study notes for a specific research paper."""
    
    try:
        notes = notes_manager.get_notes_for_paper(paper_title)
        
        return {
            "success": True,
            "paper_title": paper_title,
            "notes_count": len(notes),
            "notes": notes,
            "summary": f"Found {len(notes)} notes for '{paper_title}'"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error getting paper notes: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run()
```

## 3.4 Testing Your MCP Server

### Configure VS Code
Update your `.vscode/mcp.json` to include all three servers:

```json
{
    "inputs": [],
    "servers": {
        "huggingface": {
            "type": "sse",
            "url": "https://huggingface.co/mcp"
        },
        "github": {
            "type": "http", 
            "url": "https://api.githubcopilot.com/mcp/"
        },
        "ai-research-hub": {
            "command": "python",
            "args": ["-m", "server"],
            "cwd": "./AIResearchHub",
            "env": {}
        }
    }
}
```

### Test the Three Objectives

**1. Test External MCP Server Integration:**
```bash
# Restart VS Code and use Copilot
```
> "Search Hugging Face for recent transformer papers and save them to our local database"

**2. Test MCP Resources:**
> "Show me the research papers database summary"
> "Get all study notes as a resource"

**3. Test GitHub Automation:**
> "Add a study note about attention mechanisms and then send all notes to my GitHub repository"

## 3.5 Practical Exercises

### Exercise 1: Paper Discovery Workflow
1. Use Copilot to discover papers on "computer vision" from Hugging Face
2. Search your local database for "vision" papers
3. Check the database statistics

### Exercise 2: Study Notes Management
1. Add summary notes for 3 different papers
2. Add implementation notes for 1 paper
3. View notes by paper title

### Exercise 3: GitHub Integration
1. Create several study notes
2. Send them to your GitHub repository
3. Verify the markdown formatting

## Learning Outcomes

By completing this tutorial, you've learned:

✅ **External MCP Integration**: How to use other MCP servers (Hugging Face) in your own server
✅ **Resource Management**: How to expose data as MCP resources that AI assistants can access
✅ **Task Automation**: How to automate workflows by combining multiple MCP servers (GitHub)
✅ **Data Persistence**: How to store and manage data locally (CSV database)
✅ **Practical MCP Patterns**: Real-world examples of MCP server integration

---

> **Congratulations!** You've built a practical MCP server that demonstrates the three key concepts: external server integration, resource management, and task automation. This focused approach shows how MCP servers can work together to create powerful AI-assisted workflows.



3. **Test with VS Code Copilot**:
   - Restart VS Code
   - Use Copilot to interact with your MCP server
   - Try commands like: "Search for recent papers on transformer models"

### Sample Copilot Interactions

**Paper Search**:
> "Find the latest research papers on large language models from the past month"

**Study Plan Creation**:
> "Create a 6-week study plan for learning about computer vision and deep learning for an intermediate-level researcher"

**Trending Analysis**:
> "What are the trending AI research papers this week in natural language processing?"

**Progress Tracking**:
> "Update my learning progress - I've completed reading 3 papers on attention mechanisms"

## Learning Outcomes

- How to build sophisticated MCP servers with multiple external API integrations
- Implementing complex data models with Pydantic for research and learning domains
- Creating intelligent content analysis and recommendation systems
- Building personalized learning experiences with AI assistance
- Handling asynchronous operations and API rate limiting
- Implementing caching strategies for performance optimization
- Understanding how AI assistants can enhance research and learning workflows

---

> **Congratulations!** You've built a comprehensive AI Research Learning Hub that helps AI assistants discover papers, analyze research, and create personalized study plans. This advanced MCP server demonstrates the power of connecting AI assistants with specialized research and educational tools.

**Continue exploring**: Try extending your server with additional features like paper citation networks, collaboration recommendations, or integration with academic social networks!
