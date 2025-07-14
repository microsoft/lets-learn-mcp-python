import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from paper_manager import PaperManager

mcp = FastMCP("AI Research Learning Hub")

paper_manager = PaperManager()

@mcp.tool(
    description="Save papers from Hugging Face search results to local CSV database",
    annotations=ToolAnnotations(title="Save Papers to Database"),
)
async def save_papers_to_database(papers_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Save papers from Hugging Face MCP search results to local CSV database."""
    
    try:
        if not papers_data:
            return {
                "success": False,
                "message": "No papers data provided"
            }
        
        added_count = paper_manager.add_papers_from_huggingface(papers_data)
        stats = paper_manager.get_paper_stats()
        
        return {
            "success": True,
            "papers_provided": len(papers_data),
            "papers_saved": added_count,
            "papers_skipped": len(papers_data) - added_count,
            "database_stats": stats,
            "csv_location": str(paper_manager.csv_file),
            "summary": f"Successfully saved {added_count} papers to local database. "
                      f"Skipped {len(papers_data) - added_count} duplicates or invalid papers."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error saving papers: {str(e)}"
        }

@mcp.tool(
    description="Search local CSV database of research papers",
    annotations=ToolAnnotations(title="Local Paper Search"),
)
async def search_local_papers(query: str) -> dict[str, Any]:
    """Search the local CSV database for research papers."""
    
    try:
        papers = paper_manager.search_local_papers(query)
        stats = paper_manager.get_paper_stats()
        
        return {
            "success": True,
            "query": query,
            "results_count": len(papers),
            "papers": papers[:5], 
            "database_stats": stats,
            "summary": f"Found {len(papers)} papers matching '{query}'"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error searching papers: {str(e)}"
        }

@mcp.tool(
    description="Find GitHub repositories that implement a research paper using GitHub MCP",
    annotations=ToolAnnotations(title="Find Implementations"),
)
async def find_code_implementations_of_papers(
    query: str, language: str | None = None
) -> dict[str, Any]:
    """
    Find GitHub repositories that implement or reference a research paper
    by using the configured GitHub MCP server to search repositories and code.
    """
    try:
        # Construct optimized search queries
        repo_query = f'"{query}"'
        if language:
            repo_query += f" language:{language}"
        repo_query += " stars:>5"  # Filter for repos with some activity
        
        code_query = f'"{query}" transformer attention'
        if language:
            code_query += f" language:{language}"
        
        # Prepare the response with GitHub MCP integration info
        response = {
            "success": True,
            "query": query,
            "language_filter": language,
            "github_mcp_status": "configured",
            "search_strategy": {
                "repository_search": {
                    "query": repo_query,
                    "purpose": "Find repositories implementing the research"
                },
                "code_search": {
                    "query": code_query,
                    "purpose": "Find specific code implementations"
                }
            },
            "recommended_github_tools": [
                {
                    "tool": "mcp_github_search_repositories",
                    "query": repo_query,
                    "description": "Search for repositories by name and description"
                },
                {
                    "tool": "mcp_github_search_code", 
                    "query": code_query,
                    "description": "Search for code implementations within repositories"
                }
            ],
            "next_steps": [
                f"1. Use mcp_github_search_repositories with query: {repo_query}",
                f"2. Use mcp_github_search_code with query: {code_query}",
                "3. Examine top results for implementation quality",
                "4. Check repository stars, forks, and recent activity"
            ],
            "implementation_hints": [
                "Look for repositories with 'attention' in the name or description",
                "Check for PyTorch/TensorFlow implementations",
                "Prioritize repositories with good documentation",
                "Consider repositories with recent commits"
            ]
        }
        
        # Add language-specific suggestions
        if language:
            response["language_specific_tips"] = {
                "python": [
                    "Look for PyTorch/TensorFlow implementations",
                    "Check for Jupyter notebooks with examples",
                    "Search for 'attention_is_all_you_need' repositories"
                ],
                "javascript": [
                    "Look for TensorFlow.js implementations", 
                    "Check for web-based demos",
                    "Search for 'transformer-js' repositories"
                ]
            }.get(language.lower(), [f"Look for {language}-specific ML libraries"])
        
        return response

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error preparing GitHub search: {str(e)}",
            "fallback": "Use GitHub web interface to search manually"
        }

@mcp.resource("research://database/summary")
def get_database_summary() -> str:
    """Get a summary of the current state of the research database."""
    try:
        summary_data = {
            "papers": paper_manager.get_paper_stats(),
        }
        return json.dumps(summary_data, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.prompt(name="research_sprint")
def research_sprint_prompt(topic: str) -> str:
    return (
        f"Perform a comprehensive research sprint on '{topic}' by following these steps:\n\n"
        f"1. 📚 Use hf-mcp-server.paper_search to find papers about {topic}\n"
        f"2. 💾 Use our-research-hub.save_papers_to_database to save the top 3 papers\n"
        f"3. 🔍 Use our-research-hub.find_implementations to get optimized GitHub search strategies\n"
        f"4. 🐙 Execute GitHub searches using the configured GitHub MCP server:\n"
        f"   • mcp_github_search_repositories for repository implementations\n"
        f"   • mcp_github_search_code for specific code examples\n"
        f"5.  Summarize findings including:\n"
        f"   • Paper count and key insights\n"
        f"   • Top repositories with stars/forks\n"
        f"   • Implementation quality assessment\n"
        f"   • Code examples and demos found\n\n"
        f"💡 Pro tip: The GitHub MCP server in your mcp.json provides direct access to GitHub's API!\n"
        f"Use find_implementations for comprehensive search guidance."
    )

if __name__ == "__main__":
    mcp.run()