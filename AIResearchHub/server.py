import json

from mcp.server.fastmcp import FastMCP
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
    Generate optimized GitHub search commands for finding code implementations
    
    Args:
        topic: Research topic to find implementations for
        
    Returns:
        GitHub search strategies and commands
    """
    # Generate targeted search variations
    searches = [
        f"{topic} machine learning",
        f"{topic} python implementation", 
        f"{topic} pytorch tensorflow",
        f"{topic} algorithm code"
    ]
    
    return {
        "success": True,
        "topic": topic,
        "github_searches": searches,
        "commands": [
            f"Search repos: {topic} stars:>50",
            f"Search code: {topic} language:python"
        ],
        "instructions": "Use GitHub MCP with these search terms to find implementations"
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

@mcp.tool(description="Search local research database")
async def search_research(query: str) -> dict:
    """
    Search your local research database for matching content
    
    Args:
        query: Search term to match against research entries
        
    Returns:
        Matching research entries
    """
    papers = paper_manager.load_papers()
    matching_entries = []
    
    for entry in papers:
        if (query.lower() in entry["topic"].lower() or
            any(query.lower() in paper.get("title", "").lower() 
                for paper in entry.get("papers_found", [])) or
            any(query.lower() in repo.get("name", "").lower() 
                for repo in entry.get("repositories_found", []))):
            matching_entries.append(entry)
    
    return {
        "success": True,
        "query": query,
        "matches_found": len(matching_entries),
        "entries": matching_entries
    }

@mcp.tool(description="Update research status and add notes")
async def update_research_status(research_id: int, status: str, notes: str = "") -> dict:
    """
    Update the status of a research entry
    
    Args:
        research_id: ID of the research entry to update
        status: New status (pending, active, complete, archived)
        notes: Optional notes about the research progress
        
    Returns:
        Success status and details
    """
    valid_statuses = ["pending", "active", "complete", "archived"]
    if status not in valid_statuses:
        return {
            "success": False,
            "error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        }
    
    paper_manager.update_research_status(research_id, status, notes)
    
    return {
        "success": True,
        "research_id": research_id,
        "status": status,
        "notes": notes,
        "message": f"Research #{research_id} status updated to '{status}'"
    }

@mcp.prompt(name="research_workflow")
def research_workflow_prompt(topic: str) -> str:
    """Complete research workflow for any topic"""
    return f"""Research Topic: {topic}

WORKFLOW:
1. Use: research_topic(topic="{topic}")
   - Creates tracking entry with research ID

2. Search for papers using HuggingFace MCP tools:
   - Use: mcp_huggingface_paper_search(query="{topic}")
   - For each important paper, use: add_paper(research_id, title, authors, url)

3. Use: get_github_searches(topic="{topic}")
   - Get optimized search strategies for code implementations

4. Search for repositories using GitHub MCP tools:
   - Use: mcp_github_search_repositories(query="...")
   - For each important repo, use: add_repository(research_id, name, url, stars)

5. Use: search_research(query="{topic}")
   - Review your complete research collection

6. Use: update_research_status(research_id, "complete", "Research completed with X papers and Y repositories")
   - Mark research as complete when finished

7. Check research://status for your research dashboard

GOAL: Build comprehensive knowledge linking papers with implementations
NOTE: Remember to save findings and update status when complete!

AI RESEARCH HUB TOOLS:
- research_topic() - Start new research
- add_paper() - Save paper findings
- add_repository() - Save repo findings  
- get_github_searches() - Get search strategies
- search_research() - Search saved research
- update_research_status() - Update status

STATUS OPTIONS:
- "pending": Just started, no findings yet
- "active": Currently gathering papers/repos (auto-set when adding content)
- "complete": Research finished with comprehensive findings
- "archived": Older research for reference"""

@mcp.resource("research://status")
def research_status() -> str:
    """Current research status and saved topics"""
    summary = paper_manager.get_research_summary()
    return json.dumps(summary, indent=2)

if __name__ == "__main__":
    mcp.run()