import json
from datetime import datetime
from pathlib import Path
from typing import Any


class PaperManager:
    """Simple paper manager that matches the minimal server needs."""
    
    def __init__(self):
        """Initialize with JSON storage."""
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.papers_file = self.data_dir / "research_papers.json"
    
    def load_papers(self) -> list[dict]:
        """Load papers from JSON file."""
        if self.papers_file.exists():
            return json.loads(self.papers_file.read_text()) 
        return []
    
    def save_papers(self, papers: list[dict]):
        """Save papers to JSON file."""
        self.papers_file.write_text(json.dumps(papers, indent=2))
    
    def add_research_entry(self, topic: str) -> dict[str, Any]:
        """Add a new research entry for a topic."""
        papers = self.load_papers()
        
        research_entry = {
            "id": len(papers) + 1,
            "topic": topic,
            "created": datetime.now().isoformat(),
            "status": "pending",
            "papers_found": [],
            "repositories_found": [],
            "notes": ""
        }
        
        papers.append(research_entry)
        self.save_papers(papers)
        
        return research_entry
    
    def add_paper_to_research(self, research_id: int, paper_data: dict):
        """Add a paper to an existing research entry."""
        papers = self.load_papers()
        
        for entry in papers:
            if entry["id"] == research_id:
                entry["papers_found"].append({
                    "title": paper_data.get("title", ""),
                    "authors": paper_data.get("authors", ""),
                    "url": paper_data.get("url", "")
                })
                break
        
        self.save_papers(papers)
    
    def add_repo_to_research(self, research_id: int, repo_data: dict):
        """Add a repository to an existing research entry."""
        papers = self.load_papers()
        
        for entry in papers:
            if entry["id"] == research_id:
                entry["repositories_found"].append({
                    "name": repo_data.get("name", ""),
                    "url": repo_data.get("url", ""),
                    "stars": repo_data.get("stars", 0)
                })
                break
        
        self.save_papers(papers)
    
    def get_research_summary(self) -> dict[str, Any]:
        """Get summary of all research activities."""
        papers = self.load_papers()
        
        return {
            "total_research": len(papers),
            "entries": papers,
            "last_updated": datetime.now().isoformat()
        }
    
    def update_research_status(self, research_id: int, status: str, notes: str = ""):
        """Update the status of a research entry."""
        papers = self.load_papers()
        
        for entry in papers:
            if entry["id"] == research_id:
                entry["status"] = status
                if notes:
                    entry["notes"] = notes
                break
        
        self.save_papers(papers)