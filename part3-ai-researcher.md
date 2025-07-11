# Part 3: Building Your Own MCP Server - AI Research Learning Hub

> **Navigation**: [← Part 2: Python Study Buddy](part2-study-buddy.md) | [Back to Overview](README.md)

> **Based on**: [AIResearchLearningMCP Repository](https://github.com/jamesmontemagno/AIResearchLearningMCP)

## Overview
Learn to build an advanced MCP server from scratch by creating an AI research discovery and learning system. This tutorial will teach you the complete process of designing, implementing, and deploying a sophisticated MCP server that helps AI assistants find the latest research papers, highlight breakthroughs, and create personalized study plans.

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
Essential packages for advanced MCP server development:

```bash
pip install mcp>=0.3.0
pip install aiofiles
pip install pydantic
pip install httpx
pip install beautifulsoup4
pip install feedparser
pip install scikit-learn
pip install nltk
```

Create a **requirements.txt**:
```txt
mcp>=0.3.0
aiofiles>=23.2.0
pydantic>=2.5.0
httpx>=0.25.0
beautifulsoup4>=4.12.0
feedparser>=6.0.10
scikit-learn>=1.3.0
nltk>=3.8.1
```

### Project Structure
```bash
# Create main files
touch server.py
touch research_service.py
touch paper_analyzer.py
touch study_planner.py
touch data_models.py
touch config.py
```

Create the following directory structure:
```
AIResearchHub/
├── server.py              # Main MCP server
├── research_service.py    # Research paper discovery service
├── paper_analyzer.py      # AI paper analysis and ranking
├── study_planner.py       # Personalized study plan generator
├── data_models.py         # Pydantic models for data structures
├── config.py              # Configuration and API keys
├── requirements.txt
└── data/
    ├── papers_cache.json  # Cached research papers
    ├── user_preferences.json # Learning preferences
    └── study_plans.json   # Generated study plans
```

## 3.2 Data Models and Configuration

### Create Data Models
**Task**: Create `data_models.py` with comprehensive data structures for research papers and learning plans.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ResearchField(str, Enum):
    """Major AI/ML research fields."""
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"
    ROBOTICS = "robotics"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE_AI = "generative_ai"
    EXPLAINABLE_AI = "explainable_ai"
    AI_ETHICS = "ai_ethics"
    QUANTUM_ML = "quantum_ml"

class DifficultyLevel(str, Enum):
    """Paper difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class PaperType(str, Enum):
    """Types of research papers."""
    RESEARCH = "research"
    SURVEY = "survey"
    TUTORIAL = "tutorial"
    WORKSHOP = "workshop"
    PREPRINT = "preprint"

class Author(BaseModel):
    """Research paper author information."""
    name: str
    affiliation: Optional[str] = None
    h_index: Optional[int] = None

class ResearchPaper(BaseModel):
    """Comprehensive research paper model."""
    title: str
    authors: List[Author]
    abstract: str
    url: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    published_date: datetime
    research_fields: List[ResearchField]
    keywords: List[str]
    citation_count: Optional[int] = 0
    paper_type: PaperType
    difficulty_level: DifficultyLevel
    significance_score: float = Field(ge=0.0, le=10.0)
    summary: Optional[str] = None
    key_contributions: List[str] = []
    methodology: Optional[str] = None
    limitations: List[str] = []
    future_work: List[str] = []
    code_availability: Optional[str] = None
    dataset_availability: Optional[str] = None

class LearningGoal(BaseModel):
    """Individual learning objective."""
    title: str
    description: str
    research_fields: List[ResearchField]
    difficulty_level: DifficultyLevel
    estimated_hours: int
    prerequisites: List[str] = []
    resources: List[str] = []

class StudyPlan(BaseModel):
    """Personalized study plan."""
    plan_id: str
    user_id: str
    title: str
    description: str
    research_focus: List[ResearchField]
    difficulty_level: DifficultyLevel
    duration_weeks: int
    learning_goals: List[LearningGoal]
    recommended_papers: List[str]  # Paper IDs
    weekly_schedule: Dict[str, List[str]]  # Week -> Paper IDs
    progress_tracking: Dict[str, float] = {}  # Goal ID -> Progress (0-1)
    created_date: datetime
    last_updated: datetime

class UserPreferences(BaseModel):
    """User learning preferences and background."""
    user_id: str
    background_level: DifficultyLevel
    preferred_fields: List[ResearchField]
    time_commitment_hours_per_week: int
    learning_style: str  # "theoretical", "practical", "mixed"
    current_knowledge_areas: List[str]
    goals: List[str]
    excluded_topics: List[str] = []
```

### Create Configuration
**Task**: Create `config.py` for API keys and service configuration.

```python
import os
from pathlib import Path

class Config:
    """Configuration settings for the AI Research Hub."""
    
    # Data directories
    DATA_DIR = Path("data")
    PAPERS_CACHE_FILE = DATA_DIR / "papers_cache.json"
    USER_PREFERENCES_FILE = DATA_DIR / "user_preferences.json"
    STUDY_PLANS_FILE = DATA_DIR / "study_plans.json"
    
    # API Configuration
    ARXIV_API_BASE = "http://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
    PAPERS_WITH_CODE_API = "https://paperswithcode.com/api/v1"
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    CACHE_EXPIRY_HOURS = 24
    
    # Paper analysis settings
    MIN_CITATION_COUNT_FOR_TRENDING = 10
    SIGNIFICANCE_SCORE_THRESHOLD = 7.0
    MAX_PAPERS_PER_SEARCH = 50
    
    # Study plan settings
    DEFAULT_STUDY_DURATION_WEEKS = 8
    MAX_PAPERS_PER_WEEK = 5
    
    @classmethod
    def create_data_directories(cls):
        """Create necessary data directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
```

## 3.3 Research Service Implementation

### Create Research Discovery Service
**Task**: Create `research_service.py` that interfaces with multiple research APIs.

```python
import asyncio
import httpx
import feedparser
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET

from data_models import ResearchPaper, ResearchField, DifficultyLevel, PaperType, Author
from config import Config

class ResearchService:
    """Service for discovering and retrieving AI/ML research papers."""
    
    def __init__(self):
        self.config = Config()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache: Dict[str, Any] = self._load_cache()
        
    async def search_papers(self, 
                          query: str,
                          research_fields: Optional[List[ResearchField]] = None,
                          max_results: int = 20,
                          days_back: int = 30) -> List[ResearchPaper]:
        """Search for research papers across multiple sources."""
        
        # Check cache first
        cache_key = f"{query}_{research_fields}_{max_results}_{days_back}"
        if self._is_cache_valid(cache_key):
            return [ResearchPaper(**paper) for paper in self.cache[cache_key]]
        
        # Search across multiple sources
        papers = []
        
        # ArXiv search
        arxiv_papers = await self._search_arxiv(query, max_results // 2, days_back)
        papers.extend(arxiv_papers)
        
        # Semantic Scholar search
        scholar_papers = await self._search_semantic_scholar(query, max_results // 2)
        papers.extend(scholar_papers)
        
        # Remove duplicates and rank by significance
        papers = self._deduplicate_papers(papers)
        papers = await self._analyze_and_rank_papers(papers)
        
        # Cache results
        self.cache[cache_key] = [paper.dict() for paper in papers[:max_results]]
        self._save_cache()
        
        return papers[:max_results]
    
    async def get_trending_papers(self, 
                                research_fields: Optional[List[ResearchField]] = None,
                                time_period: str = "week") -> List[ResearchPaper]:
        """Get trending papers based on citations and social media buzz."""
        
        # Implementation for identifying trending papers
        # This would involve analyzing citation velocity, social media mentions, etc.
        pass
    
    async def _search_arxiv(self, query: str, max_results: int, days_back: int) -> List[ResearchPaper]:
        """Search ArXiv for research papers."""
        
        # Calculate date range
        start_date = datetime.now() - timedelta(days=days_back)
        date_filter = start_date.strftime("%Y%m%d")
        
        # Build ArXiv query
        search_query = f"all:{query} AND submittedDate:[{date_filter}0000 TO 999912312359]"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = await self.client.get(self.config.ARXIV_API_BASE, params=params)
            response.raise_for_status()
            
            # Parse ArXiv XML response
            papers = self._parse_arxiv_response(response.text)
            return papers
            
        except Exception as e:
            print(f"Error searching ArXiv: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[ResearchPaper]:
        """Search Semantic Scholar for research papers."""
        
        url = f"{self.config.SEMANTIC_SCHOLAR_API_BASE}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,citationCount,year,url,venue"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            papers = self._parse_semantic_scholar_response(data)
            return papers
            
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ResearchPaper]:
        """Parse ArXiv XML response into ResearchPaper objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", namespace):
                # Extract paper information from XML
                title = entry.find("atom:title", namespace).text.strip()
                abstract = entry.find("atom:summary", namespace).text.strip()
                
                # Extract authors
                authors = []
                for author in entry.findall("atom:author", namespace):
                    name = author.find("atom:name", namespace).text
                    authors.append(Author(name=name))
                
                # Extract other metadata
                published = entry.find("atom:published", namespace).text
                published_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                
                # Extract ArXiv ID and URL
                arxiv_id = entry.find("atom:id", namespace).text.split("/")[-1]
                url = f"https://arxiv.org/abs/{arxiv_id}"
                
                # Classify research field and difficulty (simplified)
                research_fields = self._classify_research_fields(title, abstract)
                difficulty_level = self._estimate_difficulty_level(abstract)
                
                paper = ResearchPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    arxiv_id=arxiv_id,
                    published_date=published_date,
                    research_fields=research_fields,
                    keywords=self._extract_keywords(abstract),
                    paper_type=PaperType.PREPRINT,
                    difficulty_level=difficulty_level,
                    significance_score=5.0  # Default, will be updated by analysis
                )
                papers.append(paper)
                
        except Exception as e:
            print(f"Error parsing ArXiv response: {e}")
        
        return papers
    
    def _classify_research_fields(self, title: str, abstract: str) -> List[ResearchField]:
        """Classify paper into research fields using keyword matching."""
        text = f"{title} {abstract}".lower()
        
        field_keywords = {
            ResearchField.MACHINE_LEARNING: ["machine learning", "ml", "learning algorithm"],
            ResearchField.DEEP_LEARNING: ["deep learning", "neural network", "cnn", "rnn", "transformer"],
            ResearchField.NATURAL_LANGUAGE_PROCESSING: ["nlp", "natural language", "text processing", "language model"],
            ResearchField.COMPUTER_VISION: ["computer vision", "image", "visual", "object detection"],
            ResearchField.REINFORCEMENT_LEARNING: ["reinforcement learning", "rl", "policy", "reward"],
            ResearchField.GENERATIVE_AI: ["generative", "gan", "diffusion", "vae", "generation"],
            # Add more field classifications
        }
        
        detected_fields = []
        for field, keywords in field_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_fields.append(field)
        
        return detected_fields if detected_fields else [ResearchField.MACHINE_LEARNING]
    
    async def _analyze_and_rank_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Analyze papers and assign significance scores."""
        # This would involve more sophisticated analysis
        # For now, we'll use a simple heuristic
        
        for paper in papers:
            significance_score = 5.0  # Base score
            
            # Adjust based on citation count
            if paper.citation_count:
                significance_score += min(paper.citation_count / 100, 3.0)
            
            # Adjust based on author reputation (simplified)
            if len(paper.authors) > 5:  # Large collaborations often significant
                significance_score += 0.5
            
            # Adjust based on recency
            days_old = (datetime.now() - paper.published_date).days
            if days_old < 7:  # Very recent
                significance_score += 1.0
            elif days_old < 30:  # Recent
                significance_score += 0.5
            
            paper.significance_score = min(significance_score, 10.0)
        
        # Sort by significance score
        papers.sort(key=lambda p: p.significance_score, reverse=True)
        return papers
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached papers from file."""
        if self.config.PAPERS_CACHE_FILE.exists():
            with open(self.config.PAPERS_CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        with open(self.config.PAPERS_CACHE_FILE, 'w') as f:
            json.dump(self.cache, f, default=str, indent=2)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache:
            return False
        
        # Check if cache is expired (simplified)
        return True  # For demo purposes
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

## 3.4 Study Plan Generation

### Create Study Plan Service
**Task**: Create `study_planner.py` that generates personalized learning roadmaps.

```python
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

from data_models import (
    StudyPlan, LearningGoal, ResearchPaper, UserPreferences, 
    ResearchField, DifficultyLevel
)
from config import Config

class StudyPlanGenerator:
    """Generates personalized study plans for AI/ML learning."""
    
    def __init__(self):
        self.config = Config()
        self.user_preferences = self._load_user_preferences()
        self.study_plans = self._load_study_plans()
    
    async def create_study_plan(self,
                              user_id: str,
                              research_focus: List[ResearchField],
                              duration_weeks: int = 8,
                              papers: Optional[List[ResearchPaper]] = None) -> StudyPlan:
        """Create a personalized study plan."""
        
        # Get user preferences
        user_prefs = self.user_preferences.get(user_id)
        if not user_prefs:
            user_prefs = UserPreferences(
                user_id=user_id,
                background_level=DifficultyLevel.INTERMEDIATE,
                preferred_fields=research_focus,
                time_commitment_hours_per_week=10,
                learning_style="mixed",
                current_knowledge_areas=[],
                goals=[]
            )
        
        # Generate learning goals
        learning_goals = self._generate_learning_goals(research_focus, user_prefs)
        
        # Select and organize papers
        if papers:
            selected_papers = self._select_papers_for_plan(papers, learning_goals, user_prefs)
        else:
            selected_papers = []
        
        # Create weekly schedule
        weekly_schedule = self._create_weekly_schedule(
            selected_papers, learning_goals, duration_weeks, user_prefs
        )
        
        # Create study plan
        plan = StudyPlan(
            plan_id=str(uuid.uuid4()),
            user_id=user_id,
            title=f"AI/ML Learning Plan: {', '.join([f.value for f in research_focus])}",
            description=f"Personalized {duration_weeks}-week study plan covering {', '.join([f.value for f in research_focus])}",
            research_focus=research_focus,
            difficulty_level=user_prefs.background_level,
            duration_weeks=duration_weeks,
            learning_goals=learning_goals,
            recommended_papers=[paper.arxiv_id or paper.url for paper in selected_papers],
            weekly_schedule=weekly_schedule,
            created_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Save study plan
        self.study_plans[plan.plan_id] = plan.dict()
        self._save_study_plans()
        
        return plan
    
    def _generate_learning_goals(self, 
                               research_fields: List[ResearchField],
                               user_prefs: UserPreferences) -> List[LearningGoal]:
        """Generate learning goals based on research fields and user preferences."""
        
        goals = []
        
        # Define goal templates for different research fields
        goal_templates = {
            ResearchField.DEEP_LEARNING: [
                LearningGoal(
                    title="Understanding Neural Network Architectures",
                    description="Learn fundamental and advanced neural network architectures",
                    research_fields=[ResearchField.DEEP_LEARNING],
                    difficulty_level=DifficultyLevel.INTERMEDIATE,
                    estimated_hours=15,
                    prerequisites=["Basic machine learning", "Linear algebra"],
                    resources=["Research papers", "Implementation tutorials"]
                ),
                LearningGoal(
                    title="Transformer Models and Attention Mechanisms",
                    description="Master transformer architecture and attention mechanisms",
                    research_fields=[ResearchField.DEEP_LEARNING, ResearchField.NATURAL_LANGUAGE_PROCESSING],
                    difficulty_level=DifficultyLevel.ADVANCED,
                    estimated_hours=20,
                    prerequisites=["Neural networks", "Sequence modeling"],
                    resources=["Attention is All You Need paper", "Implementation guides"]
                )
            ],
            ResearchField.NATURAL_LANGUAGE_PROCESSING: [
                LearningGoal(
                    title="Large Language Models",
                    description="Understand the architecture and training of large language models",
                    research_fields=[ResearchField.NATURAL_LANGUAGE_PROCESSING],
                    difficulty_level=DifficultyLevel.ADVANCED,
                    estimated_hours=25,
                    prerequisites=["Transformers", "Deep learning"],
                    resources=["GPT papers", "BERT implementation"]
                )
            ],
            # Add more field-specific goals
        }
        
        # Select goals based on research fields and user level
        for field in research_fields:
            if field in goal_templates:
                field_goals = goal_templates[field]
                # Filter goals by user's difficulty level
                suitable_goals = [
                    goal for goal in field_goals 
                    if self._is_goal_suitable(goal, user_prefs)
                ]
                goals.extend(suitable_goals)
        
        return goals
    
    def _select_papers_for_plan(self,
                              papers: List[ResearchPaper],
                              learning_goals: List[LearningGoal],
                              user_prefs: UserPreferences) -> List[ResearchPaper]:
        """Select papers that align with learning goals and user preferences."""
        
        selected_papers = []
        
        # Group papers by research field
        papers_by_field = {}
        for paper in papers:
            for field in paper.research_fields:
                if field not in papers_by_field:
                    papers_by_field[field] = []
                papers_by_field[field].append(paper)
        
        # Select papers for each learning goal
        for goal in learning_goals:
            goal_papers = []
            for field in goal.research_fields:
                if field in papers_by_field:
                    # Select papers that match difficulty level and significance
                    suitable_papers = [
                        paper for paper in papers_by_field[field]
                        if self._is_paper_suitable(paper, goal, user_prefs)
                    ]
                    # Sort by significance and take top papers
                    suitable_papers.sort(key=lambda p: p.significance_score, reverse=True)
                    goal_papers.extend(suitable_papers[:3])  # Max 3 papers per goal per field
            
            selected_papers.extend(goal_papers)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_papers = []
        for paper in selected_papers:
            if paper.url not in seen:
                seen.add(paper.url)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _create_weekly_schedule(self,
                              papers: List[ResearchPaper],
                              goals: List[LearningGoal],
                              duration_weeks: int,
                              user_prefs: UserPreferences) -> Dict[str, List[str]]:
        """Create a weekly schedule for the study plan."""
        
        schedule = {}
        papers_per_week = max(1, len(papers) // duration_weeks)
        
        # Distribute papers across weeks
        for week in range(1, duration_weeks + 1):
            week_key = f"week_{week}"
            start_idx = (week - 1) * papers_per_week
            end_idx = start_idx + papers_per_week
            
            week_papers = papers[start_idx:end_idx]
            schedule[week_key] = [paper.url for paper in week_papers]
        
        # Add any remaining papers to the last week
        remaining_papers = papers[duration_weeks * papers_per_week:]
        if remaining_papers:
            last_week = f"week_{duration_weeks}"
            schedule[last_week].extend([paper.url for paper in remaining_papers])
        
        return schedule
    
    def _is_goal_suitable(self, goal: LearningGoal, user_prefs: UserPreferences) -> bool:
        """Check if a learning goal is suitable for the user."""
        # Check difficulty level
        difficulty_order = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, 
                          DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
        
        user_level_idx = difficulty_order.index(user_prefs.background_level)
        goal_level_idx = difficulty_order.index(goal.difficulty_level)
        
        # Allow goals up to one level above user's current level
        return goal_level_idx <= user_level_idx + 1
    
    def _is_paper_suitable(self, paper: ResearchPaper, goal: LearningGoal, user_prefs: UserPreferences) -> bool:
        """Check if a paper is suitable for a learning goal and user."""
        # Check if paper's difficulty matches goal's difficulty
        if paper.difficulty_level != goal.difficulty_level:
            return False
        
        # Check significance threshold
        if paper.significance_score < self.config.SIGNIFICANCE_SCORE_THRESHOLD:
            return False
        
        # Check if paper is recent enough (within last 2 years for cutting-edge research)
        if (datetime.now() - paper.published_date).days > 730:
            return False
        
        return True
    
    def _load_user_preferences(self) -> Dict[str, UserPreferences]:
        """Load user preferences from file."""
        if self.config.USER_PREFERENCES_FILE.exists():
            with open(self.config.USER_PREFERENCES_FILE, 'r') as f:
                data = json.load(f)
                return {k: UserPreferences(**v) for k, v in data.items()}
        return {}
    
    def _load_study_plans(self) -> Dict[str, Dict]:
        """Load study plans from file."""
        if self.config.STUDY_PLANS_FILE.exists():
            with open(self.config.STUDY_PLANS_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_study_plans(self) -> None:
        """Save study plans to file."""
        with open(self.config.STUDY_PLANS_FILE, 'w') as f:
            json.dump(self.study_plans, f, default=str, indent=2)
```

## 3.5 MCP Server Implementation

### Create the Main MCP Server
**Task**: Create `server.py` that implements the MCP server with all tools and resources.

```python
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from research_service import ResearchService
from study_planner import StudyPlanGenerator
from data_models import ResearchField, DifficultyLevel, StudyPlan
from config import Config

# Initialize the MCP server
mcp = FastMCP("AI Research Learning Hub")

# Initialize services
research_service = ResearchService()
study_planner = StudyPlanGenerator()
config = Config()

# Create data directories
config.create_data_directories()

@mcp.tool(
    description="Search for the latest AI/ML research papers by topic and research field",
    annotations=ToolAnnotations(title="Research Paper Search", idempotentHint=True, readOnlyHint=True),
)
async def search_research_papers(
    query: str,
    research_fields: Optional[List[str]] = None,
    max_results: int = 20,
    days_back: int = 30
) -> Dict[str, Any]:
    """Search for AI/ML research papers with advanced filtering options."""
    
    try:
        # Convert string fields to ResearchField enums
        fields = []
        if research_fields:
            for field_str in research_fields:
                try:
                    fields.append(ResearchField(field_str.lower()))
                except ValueError:
                    continue
        
        # Search for papers
        papers = await research_service.search_papers(
            query=query,
            research_fields=fields if fields else None,
            max_results=max_results,
            days_back=days_back
        )
        
        # Convert to serializable format
        papers_data = []
        for paper in papers:
            paper_dict = paper.dict()
            paper_dict['published_date'] = paper.published_date.isoformat()
            paper_dict['research_fields'] = [field.value for field in paper.research_fields]
            papers_data.append(paper_dict)
        
        return {
            "success": True,
            "query": query,
            "research_fields": research_fields,
            "results_count": len(papers_data),
            "papers": papers_data,
            "summary": f"Found {len(papers_data)} research papers on '{query}' from the last {days_back} days"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error searching for papers: {str(e)}"
        }

@mcp.tool(
    description="Get trending AI/ML research papers based on citations and buzz",
    annotations=ToolAnnotations(title="Trending Papers Discovery", idempotentHint=True, readOnlyHint=True),
)
async def get_trending_papers(
    research_fields: Optional[List[str]] = None,
    time_period: str = "week",
    max_results: int = 15
) -> Dict[str, Any]:
    """Discover trending and breakthrough AI/ML research papers."""
    
    try:
        # Convert string fields to ResearchField enums
        fields = []
        if research_fields:
            for field_str in research_fields:
                try:
                    fields.append(ResearchField(field_str.lower()))
                except ValueError:
                    continue
        
        # Get trending papers
        papers = await research_service.get_trending_papers(
            research_fields=fields if fields else None,
            time_period=time_period
        )
        
        # For demo, we'll use the search results and filter by significance
        if not papers:
            # Fallback to high-significance papers from recent search
            papers = await research_service.search_papers(
                query="artificial intelligence machine learning",
                research_fields=fields if fields else None,
                max_results=max_results,
                days_back=7
            )
            # Filter for high significance
            papers = [p for p in papers if p.significance_score >= 7.0]
        
        # Convert to serializable format
        papers_data = []
        for paper in papers[:max_results]:
            paper_dict = paper.dict()
            paper_dict['published_date'] = paper.published_date.isoformat()
            paper_dict['research_fields'] = [field.value for field in paper.research_fields]
            papers_data.append(paper_dict)
        
        return {
            "success": True,
            "time_period": time_period,
            "research_fields": research_fields,
            "results_count": len(papers_data),
            "trending_papers": papers_data,
            "summary": f"Found {len(papers_data)} trending papers in {time_period}",
            "trend_analysis": {
                "avg_significance_score": sum(p["significance_score"] for p in papers_data) / len(papers_data) if papers_data else 0,
                "top_research_fields": list(set([field for paper in papers_data for field in paper["research_fields"]])),
                "citation_range": {
                    "min": min((p.get("citation_count", 0) for p in papers_data), default=0),
                    "max": max((p.get("citation_count", 0) for p in papers_data), default=0)
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error getting trending papers: {str(e)}"
        }

@mcp.tool(
    description="Create a personalized study plan for learning AI/ML research topics",
    annotations=ToolAnnotations(title="Study Plan Generator", idempotentHint=False, readOnlyHint=False),
)
async def create_study_plan(
    user_id: str,
    research_focus: List[str],
    duration_weeks: int = 8,
    background_level: str = "intermediate",
    time_commitment_hours: int = 10
) -> Dict[str, Any]:
    """Generate a personalized study plan for AI/ML research learning."""
    
    try:
        # Convert string inputs to enums
        focus_fields = []
        for field_str in research_focus:
            try:
                focus_fields.append(ResearchField(field_str.lower()))
            except ValueError:
                continue
        
        if not focus_fields:
            return {
                "success": False,
                "error": "No valid research fields provided",
                "available_fields": [field.value for field in ResearchField]
            }
        
        # Get recent papers for the focus areas
        papers = await research_service.search_papers(
            query=" ".join([field.value for field in focus_fields]),
            research_fields=focus_fields,
            max_results=50,
            days_back=90
        )
        
        # Create study plan
        study_plan = await study_planner.create_study_plan(
            user_id=user_id,
            research_focus=focus_fields,
            duration_weeks=duration_weeks,
            papers=papers
        )
        
        # Convert to serializable format
        plan_dict = study_plan.dict()
        plan_dict['created_date'] = study_plan.created_date.isoformat()
        plan_dict['last_updated'] = study_plan.last_updated.isoformat()
        plan_dict['research_focus'] = [field.value for field in study_plan.research_focus]
        plan_dict['difficulty_level'] = study_plan.difficulty_level.value
        
        return {
            "success": True,
            "study_plan": plan_dict,
            "recommendations": {
                "total_papers": len(study_plan.recommended_papers),
                "total_learning_goals": len(study_plan.learning_goals),
                "estimated_total_hours": sum(goal.estimated_hours for goal in study_plan.learning_goals),
                "papers_per_week": len(study_plan.recommended_papers) // duration_weeks if duration_weeks > 0 else 0
            },
            "summary": f"Created {duration_weeks}-week study plan covering {', '.join([field.value for field in focus_fields])}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error creating study plan: {str(e)}"
        }

@mcp.tool(
    description="Summarize and analyze a research paper for key insights",
    annotations=ToolAnnotations(title="Paper Analyzer", idempotentHint=True, readOnlyHint=True),
)
async def summarize_paper(
    paper_url: str,
    analysis_depth: str = "standard"
) -> Dict[str, Any]:
    """Analyze and summarize a research paper for key insights and contributions."""
    
    try:
        # This would involve fetching the paper content and analyzing it
        # For demo purposes, we'll provide a structured analysis template
        
        analysis = {
            "paper_url": paper_url,
            "analysis_depth": analysis_depth,
            "summary": {
                "main_contribution": "Key contribution analysis would go here",
                "methodology": "Methodology summary would be extracted",
                "key_findings": [
                    "Key finding 1",
                    "Key finding 2", 
                    "Key finding 3"
                ],
                "limitations": [
                    "Limitation 1",
                    "Limitation 2"
                ],
                "future_work": [
                    "Future research direction 1",
                    "Future research direction 2"
                ]
            },
            "technical_details": {
                "datasets_used": ["Dataset 1", "Dataset 2"],
                "metrics": ["Accuracy", "F1-Score", "BLEU"],
                "baseline_comparisons": ["Baseline 1", "Baseline 2"],
                "code_availability": "Check paper for implementation details"
            },
            "relevance_analysis": {
                "impact_score": 8.5,
                "practical_applications": ["Application 1", "Application 2"],
                "related_work": ["Related paper 1", "Related paper 2"],
                "follow_up_recommendations": [
                    "Read the original transformer paper for background",
                    "Implement the proposed method",
                    "Compare with recent approaches"
                ]
            }
        }
        
        return {
            "success": True,
            "paper_analysis": analysis,
            "summary": f"Completed {analysis_depth} analysis of research paper"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error analyzing paper: {str(e)}"
        }

@mcp.tool(
    description="Track learning progress and update study plan milestones",
    annotations=ToolAnnotations(title="Progress Tracker", idempotentHint=False, readOnlyHint=False),
)
async def track_learning_progress(
    user_id: str,
    study_plan_id: str,
    completed_papers: List[str],
    learning_notes: str = "",
    difficulty_rating: float = 5.0
) -> Dict[str, Any]:
    """Track learning progress and update study plan completion status."""
    
    try:
        # Update progress tracking
        progress_update = {
            "user_id": user_id,
            "study_plan_id": study_plan_id,
            "completed_papers": completed_papers,
            "learning_notes": learning_notes,
            "difficulty_rating": difficulty_rating,
            "timestamp": datetime.now().isoformat(),
            "completion_percentage": 0.0  # Would calculate based on actual progress
        }
        
        # Calculate progress metrics
        total_papers = 20  # Would get from actual study plan
        completion_percentage = (len(completed_papers) / total_papers) * 100 if total_papers > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if completion_percentage < 25:
            recommendations.append("Focus on foundational papers first")
        elif completion_percentage < 75:
            recommendations.append("Consider exploring related research areas")
        else:
            recommendations.append("Prepare for advanced topics in your next study plan")
        
        return {
            "success": True,
            "progress_update": progress_update,
            "metrics": {
                "completion_percentage": completion_percentage,
                "papers_completed": len(completed_papers),
                "average_difficulty_rating": difficulty_rating,
                "study_streak_days": 5  # Would track actual streak
            },
            "recommendations": recommendations,
            "achievements": [
                "First Paper Completed",
                "Weekly Goal Achieved"
            ] if len(completed_papers) > 0 else [],
            "summary": f"Updated progress for study plan {study_plan_id}: {completion_percentage:.1f}% complete"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error tracking progress: {str(e)}"
        }

@mcp.resource("ai-research://fields/available")
def get_research_fields() -> str:
    """Get all available AI/ML research fields."""
    fields = {
        "research_fields": [
            {
                "name": field.value,
                "description": f"Research in {field.value.replace('_', ' ').title()}",
                "example_topics": []  # Would populate with actual examples
            }
            for field in ResearchField
        ]
    }
    return json.dumps(fields, indent=2)

@mcp.resource("ai-research://difficulty-levels")
def get_difficulty_levels() -> str:
    """Get information about difficulty levels for learning content."""
    levels = {
        "difficulty_levels": [
            {
                "level": level.value,
                "description": f"{level.value.title()} level content",
                "typical_prerequisites": [],  # Would populate with actual prerequisites
                "time_commitment": "Varies"
            }
            for level in DifficultyLevel
        ]
    }
    return json.dumps(levels, indent=2)

@mcp.resource("ai-research://study-tips")
def get_study_tips() -> str:
    """Get tips and best practices for studying AI/ML research."""
    tips = {
        "study_tips": {
            "reading_papers": [
                "Start with the abstract and conclusion",
                "Focus on understanding the main contribution",
                "Take notes on methodology and results",
                "Look up unfamiliar terms and concepts",
                "Try to implement key algorithms when possible"
            ],
            "staying_current": [
                "Follow top conferences (NeurIPS, ICML, ICLR)",
                "Subscribe to arXiv feeds in your areas of interest",
                "Join academic Twitter and follow researchers",
                "Participate in reading groups and discussion forums",
                "Attend virtual conferences and workshops"
            ],
            "building_understanding": [
                "Connect new papers to your existing knowledge",
                "Discuss papers with peers and mentors",
                "Write summaries in your own words",
                "Identify gaps between theory and practice",
                "Experiment with code implementations"
            ]
        }
    }
    return json.dumps(tips, indent=2)

# Cleanup function to close resources
async def cleanup():
    """Clean up resources when shutting down."""
    await research_service.close()

if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        asyncio.run(cleanup())
````

## 3.6 Testing and Deployment

### Test Your MCP Server

1. **Install the server**:
```bash
pip install -e .
```

2. **Configure VS Code MCP settings** in `.vscode/mcp.json`:
```json
{
    "inputs": [],
    "servers": {
        "ai-research-hub": {
            "command": "python",
            "args": ["-m", "server"],
            "cwd": "./AIResearchHub",
            "env": {}
        }
    }
}
```

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

