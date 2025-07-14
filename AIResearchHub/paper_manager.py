import csv
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd


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
    
    def add_papers_from_huggingface(self, papers_data: list[dict[str, Any]]) -> int:
        """Add papers from Hugging Face search to local CSV database."""
        if not papers_data:
            return 0
            
        # Get existing paper IDs to avoid duplicates
        existing_papers = set()
        df = self.get_papers_dataframe()
        if not df.empty:
            existing_papers = set(df['paper_id'].astype(str))
        
        added_count = 0
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for paper in papers_data:
                # Validate required fields
                if not paper.get('id') or not paper.get('title'):
                    continue
                    
                paper_id = str(paper.get('id', ''))
                
                # Skip if paper already exists
                if paper_id in existing_papers:
                    continue
                
                # Handle authors properly - could be list of dicts or strings
                authors = paper.get('authors', [])
                if isinstance(authors, list):
                    if authors and isinstance(authors[0], dict):
                        author_names = [author.get('name', '') for author in authors]
                    else:
                        author_names = [str(author) for author in authors]
                    authors_str = ', '.join(filter(None, author_names))
                else:
                    authors_str = str(authors)
                
                # Handle keywords properly
                keywords = paper.get('keywords', [])
                if isinstance(keywords, list):
                    keywords_str = ', '.join(filter(None, [str(k) for k in keywords]))
                else:
                    keywords_str = str(keywords)
                
                paper_row = [
                    paper_id,
                    paper.get('title', ''),
                    authors_str,
                    paper.get('abstract', ''),
                    paper.get('url', ''),
                    paper.get('published_date', ''),
                    paper.get('research_field', 'machine_learning'),
                    keywords_str,
                    int(paper.get('downloads', 0)),
                    int(paper.get('likes', 0)),
                    datetime.now().isoformat()
                ]
                
                writer.writerow(paper_row)
                existing_papers.add(paper_id)  # Add to set to prevent duplicates in same batch
                added_count += 1
        
        return added_count
    
    def get_papers_dataframe(self) -> pd.DataFrame:
        """Get all papers as a pandas DataFrame."""
        if self.csv_file.exists():
            return pd.read_csv(self.csv_file)
        return pd.DataFrame()
    
    def search_local_papers(self, query: str) -> list[dict[str, Any]]:
        """Search local CSV database for papers."""
        df = self.get_papers_dataframe()
        if df.empty:
            return []
        
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Simple text search in title, abstract, and keywords
        mask = (
            df['title'].str.lower().str.contains(query_lower, case=False, na=False) |
            df['abstract'].str.lower().str.contains(query_lower, case=False, na=False) |
            df['keywords'].str.lower().str.contains(query_lower, case=False, na=False) |
            df['authors'].str.lower().str.contains(query_lower, case=False, na=False)
        )
        
        matching_papers = df[mask]
        
        # Sort by most recent first, then by likes
        if not matching_papers.empty:
            matching_papers = matching_papers.sort_values(
                ['published_date', 'likes'], 
                ascending=[False, False]
            )
        
        return cast(list[dict[str, Any]], matching_papers.to_dict('records'))
    
    def get_paper_stats(self) -> dict[str, Any]:
        """Get statistics about the local paper database."""
        df = self.get_papers_dataframe()
        if df.empty:
            return {"total_papers": 0}
        
        return {
            "total_papers": len(df),
            "research_fields": df['research_field'].value_counts().to_dict(),
            "most_recent_paper": df['published_date'].max() if 'published_date' in df.columns else None,
            "most_liked_paper": df.loc[df['likes'].idxmax(
                ), 'title'] if 'likes' in df.columns and not df.empty else None,
            "database_last_updated": df['added_date'].max() if 'added_date' in df.columns else None
        }
    
    def get_paper_by_id(self, paper_id: str) -> dict[str, Any] | None:
        """Get a specific paper by its ID."""
        df = self.get_papers_dataframe()
        if df.empty:
            return None
        
        matching_papers = df[df['paper_id'].astype(str) == str(paper_id)]
        if matching_papers.empty:
            return None
        
        return cast(dict[str, Any], matching_papers.iloc[0].to_dict())
    
    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper already exists in the database."""
        return self.get_paper_by_id(paper_id) is not None
    
    def remove_paper(self, paper_id: str) -> bool:
        """Remove a paper from the database."""
        df = self.get_papers_dataframe()
        if df.empty:
            return False
        
        # Filter out the paper to remove
        df_filtered = df[df['paper_id'].astype(str) != str(paper_id)]
        
        # Check if anything was actually removed
        if len(df_filtered) == len(df):
            return False
        
        # Write back to CSV
        df_filtered.to_csv(self.csv_file, index=False)
        return True