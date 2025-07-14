import json

from fastmcp import FastMCP
from paper_manager import PaperManager

# Initialize MCP server and paper manager
mcp = FastMCP("AI Research Hub")
paper_manager = PaperManager()

@mcp.tool(description="Start researching a topic and get research ID")
async def research_topic(topic: str) -> dict:
    """
    Create a new research entry for tracking papers and implementations
    
    Args:
        topic: Research topic to investigate
        
    Returns:
        Research entry details with tracking ID
    """
    research_entry = paper_manager.add_research_entry(topic)
    
    return {
        "success": True,
        "topic": topic,
        "research_id": research_entry["id"],
        "message": f"Research entry #{research_entry['id']} created for '{topic}'",
        "total_research_topics": len(paper_manager.load_papers())
    }

@mcp.tool(description="Get GitHub search strategies for finding implementations") 
async def get_github_searches(topic: str) -> dict:
    """
    Generate GitHub search commands for finding code implementations
    
    Args:
        topic: Research topic to find implementations for
        
    Returns:
        GitHub search strategies and commands
    """
    return {
        "success": True,
        "topic": topic,
        "github_searches": [f"{topic} implementation", f"{topic} python"],
        "commands": [f"Search repos: {topic} stars:>10"],
        "instructions": "Use GitHub MCP with these search terms"
    }

@mcp.tool(description="Add a paper to a research entry")
async def add_paper(research_id: int, title: str, authors: str = "", url: str = "") -> dict:
    """
    Add a research paper to an existing research entry
    
    Args:
        research_id: ID of the research entry to add to
        title: Title of the paper
        authors: Paper authors (optional)
        url: URL to the paper (optional)
        
    Returns:
        Success status and details
    """
    paper_data = {
        "title": title,
        "authors": authors,
        "url": url
    }
    
    paper_manager.add_paper_to_research(research_id, paper_data)
    
    return {
        "success": True,
        "research_id": research_id,
        "paper_added": title,
        "message": f"Paper '{title}' added to research #{research_id}"
    }

@mcp.tool(description="Add a repository to a research entry")
async def add_repository(research_id: int, name: str, url: str = "", stars: int = 0) -> dict:
    """
    Add a code repository to an existing research entry
    
    Args:
        research_id: ID of the research entry to add to
        name: Repository name
        url: Repository URL (optional)
        stars: Star count (optional)
        
    Returns:
        Success status and details
    """
    repo_data = {
        "name": name,
        "url": url,
        "stars": stars
    }
    
    paper_manager.add_repo_to_research(research_id, repo_data)
    
    return {
        "success": True,
        "research_id": research_id,
        "repository_added": name,
        "message": f"Repository '{name}' added to research #{research_id}"
    }

@mcp.tool(description="Update research status and add notes")
async def update_research_status(research_id: int, status: str, notes: str = "") -> dict:
    """
    Update the status of a research entry
    
    Args:
        research_id: ID of the research entry to update
        status: New status (pending, active, complete)
        notes: Optional notes about the research progress
        
    Returns:
        Success status and details
    """
    paper_manager.update_research_status(research_id, status, notes)
    
    return {
        "success": True,
        "research_id": research_id,
        "status": status,
        "message": f"Research #{research_id} status updated to '{status}'"
    }

@mcp.prompt(name="research_workflow")
def research_workflow_prompt(topic: str) -> str:
    """Complete research workflow for any topic"""
    return f"""Research Topic: {topic}

WORKFLOW:
1. research_topic(topic="{topic}") - Create research entry
2. Search HuggingFace papers: mcp_huggingface_paper_search(query="{topic}")
3. get_github_searches(topic="{topic}") - Get search strategies  
4. Search GitHub repos: mcp_github_search_repositories(query="{topic}")
5. Add findings: add_paper() and add_repository()
6. update_research_status(research_id, "complete") - Mark done
7. Check research://status for dashboard

TOOLS: research_topic, add_paper, add_repository, get_github_searches, update_research_status
GOAL: Link papers with implementations"""

@mcp.resource("status://dashboard")
def research_status() -> str:
    """Current research status and saved topics"""
    summary = paper_manager.get_research_summary()
    return json.dumps(summary, indent=2)

if __name__ == "__main__":
    mcp.run()