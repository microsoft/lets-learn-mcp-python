# Part 2: Using MCP Servers - Building Python Study Buddy App

> **Navigation**: [← Part 1: Setup](part1-setup.md) | [Next: Part 3 - AI Research Server →](part3-ai-research-server.md)

> **Based on**: [PythonStudyBuddyMCP Repository](https://github.com/jamesmontemagno/PythonStudyBuddyMCP)

## Overview
In this section, you'll learn how to use MCP servers by building an interactive Python learning companion that helps developers at different skill levels master Python concepts. This hands-on tutorial will teach you how MCP servers work from a consumer perspective before you build your own advanced server.

## Step-by-Step Walkthrough

### 2.1 Project Setup

#### Create a new Python project
- Create a new folder called **PythonStudyBuddy** and then create your Python project inside:

```bash
mkdir PythonStudyBuddy
cd PythonStudyBuddy
```

#### Set up virtual environment
```bash
# Create virtual environment
python -m venv study-env

# Activate on macOS/Linux
source study-env/bin/activate

# Activate on Windows
study-env\Scripts\activate

# Install dependencies
pip install mcp asyncio dataclasses-json
```

#### Create project structure
```bash
# Create main files
touch main.py
touch study_session.py
touch learning_concepts.py
touch progress_tracker.py
touch requirements.txt
```

#### Create requirements.txt
```txt
mcp>=0.3.0
asyncio
dataclasses-json
colorama
```

#### Push to GitHub
- Create a new GitHub repository
- Connect your local project
- Make your first commit
- This can be done through the VS Code Source Control panel
- Ensure your repository is public so Copilot can access it

### 2.2 MCP Server Configuration and Initial Setup

#### Configure Python Learning MCP
- Install and configure a Python Learning MCP server in VS Code
- Understanding the Learning API endpoints
- Testing the MCP server connection
- Verify you can retrieve learning concepts through Copilot

Create a new folder named **.vscode** and create a file named **mcp.json** with the following content:

```json
"learnpython-mcp": {
         "command": "/opt/anaconda3/bin/uv",
         "args": [
            "--directory",
            ".",
            "run",
            "server.py"
         ]
      }
```

#### Configure GitHub MCP
- Install GitHub MCP server in VS Code for progress tracking
- Set up GitHub authentication tokens
- Configure repository access permissions
- Verify GitHub MCP functionality for creating study milestone issues

Update the **mcp.json** with a new server, this time will be a remote server for GitHub:

```json
"github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/"
    }
```

Your final **mcp.json** should look like this:

```json
{
    "inputs": [],
    "servers": {
        "python-learning": {
            "command": "python",
            "args": [
                "-m", "python_learning_server"
            ],
            "env": {}
        },
        "github": {
            "type": "http",
            "url": "https://api.githubcopilot.com/mcp/"
        }
    }
}
```

#### Setup Copilot Instructions
After creating your repository and pushing the initial commit, let's set up Copilot with project-specific context.

**Task**: Create `.github/copilot-instructions.md` file
**Steps**:
1. Create `.github` folder in your repository root
2. Add `copilot-instructions.md` file with the following content:

```markdown
This project is Python 3.12+ and uses modern Python features including dataclasses and asyncio.

Make sure all code generated is inside of the PythonStudyBuddy project, which may be a subfolder inside of the main folder.

It is on GitHub at https://github.com/YOUR_USERNAME/YOUR_REPO_NAME

## Project Context
This is an interactive console application that helps developers learn Python concepts at beginner, intermediate, and expert levels through personalized study sessions.

## Python Coding Standards
- Use snake_case for function names, method names, and variables
- Use PascalCase for class names
- Use descriptive names that clearly indicate purpose
- Add docstrings for public functions and classes
- Use type hints for function parameters and return values
- Prefer f-strings for string formatting
- Use async/await for asynchronous operations
- Follow the repository pattern for data access
- Use proper exception handling with try-except blocks
- Use dataclasses for data models
- Use pathlib for file operations
- Follow PEP 8 style guidelines

## Naming Conventions
- Classes: `StudySession`, `LearningConcept`, `ProgressTracker`
- Functions: `get_concepts_by_level()`, `start_study_session()`, `generate_challenge()`
- Properties: `name`, `level`, `difficulty`, `mastery_score`
- Variables: `current_concept`, `study_progress`, `user_response`
- Constants: `BEGINNER_LEVEL`, `INTERMEDIATE_LEVEL`, `EXPERT_LEVEL`

## Architecture
- Interactive console application with study session management
- Helper classes for learning concept management
- Dataclass models for learning data representation
- Progress tracking and gamification elements
- Separation of concerns between UI and learning logic
```

**Important**: Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

#### Create GitHub Issue for Project Planning
Before we start coding, let's use the GitHub MCP server to create an issue that outlines our study buddy goals.

**Task**: Use VS Code with GitHub MCP to create a new issue
**Sample Copilot Command**:
> "Create a new GitHub issue in my repository titled 'Implement Python Study Buddy Learning Application' with the following requirements: Create an interactive Python console app that provides personalized learning sessions for developers at beginner, intermediate, and expert levels. Include concept explanations, coding challenges, progress tracking, and achievement badges. Add appropriate labels like 'enhancement', 'educational', and 'good first issue'. Include a detailed implementation checklist."

### 2.3 Interactive Development with Copilot

#### Create Learning Concept Data Model
- **Task**: Ask Copilot to create a Python dataclass model for learning concepts
- **Expected Output**: A `learning_concepts.py` file with the following structure:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

class SkillLevel(Enum):
    """Enumeration for Python skill levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

class ConceptType(Enum):
    """Types of learning concepts."""
    SYNTAX = "syntax"
    DATA_STRUCTURES = "data_structures"
    ALGORITHMS = "algorithms"
    OOP = "object_oriented_programming"
    FUNCTIONAL = "functional_programming"
    ASYNC = "asynchronous_programming"
    TESTING = "testing"
    PERFORMANCE = "performance"

@dataclass
class LearningConcept:
    """Represents a Python learning concept."""
    name: str
    level: SkillLevel
    concept_type: ConceptType
    description: str
    explanation: str
    code_examples: List[str]
    practice_challenges: List[str]
    prerequisites: List[str]
    related_concepts: List[str]
    mastery_criteria: List[str]
    estimated_time_minutes: int
    difficulty_score: float  # 1.0-10.0
    
    def __str__(self) -> str:
        """Return a formatted string representation of the concept."""
        return f"{self.name} ({self.level.value}) - {self.concept_type.value}"

@dataclass
class StudyProgress:
    """Tracks progress on learning concepts."""
    concept_name: str
    mastery_score: float  # 0.0-1.0
    attempts: int
    time_spent_minutes: int
    last_studied: str  # ISO format date
    achievements: List[str]
    notes: str = ""
```

#### Build the Study Session Manager
- **Task**: Create an application that manages interactive learning sessions
- **Expected Features**:
  - Console application that presents personalized study sessions
  - Show concept explanations with code examples
  - Generate coding challenges based on skill level
  - Track progress and provide encouraging feedback
  - Achievement badges and milestone celebrations

**Sample Output**:
```
🐍 Python Study Buddy - Interactive Learning Session
===================================================
Level: Intermediate
Progress: 3/10 concepts mastered this week

📚 Today's Focus: List Comprehensions
🎯 Difficulty: 6.5/10 | Estimated Time: 25 minutes

📖 Concept Overview:
List comprehensions provide a concise way to create lists based on existing 
iterables with optional filtering and transformation logic.

💡 Code Example:
# Create a list of even squares
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# Result: [0, 4, 16, 36, 64]

🎯 Your Challenge:
Write a list comprehension that creates a list of all words in a sentence 
that are longer than 3 characters, converted to uppercase.

sentence = "The quick brown fox jumps over the lazy dog"
# Your code here: result = [...]

Type your answer (or 'hint' for help, 'skip' to move on):
```

#### Create Learning Concept Database
- **Task**: Ask Copilot to create a comprehensive database of Python concepts
- **Expected Output**: A structured collection covering beginner to expert levels

**Sample Copilot Command**:
> "Create a comprehensive database of Python learning concepts organized by skill level. Include 10+ concepts for each level (beginner, intermediate, expert) covering syntax, data structures, OOP, async programming, testing, and performance. Each concept should have explanations, code examples, practice challenges, and mastery criteria."

**Expected Beginner Concepts**:
- Variables and Data Types
- Control Flow (if/else, loops)
- Functions and Parameters
- Lists and Dictionaries
- String Manipulation
- File I/O Basics
- Error Handling (try/except)
- Basic Classes and Objects
- Modules and Imports
- List and Dictionary Methods

**Expected Intermediate Concepts**:
- List/Dict Comprehensions
- Decorators
- Context Managers
- Generators and Iterators
- Lambda Functions
- Regular Expressions
- JSON and API Handling
- Unit Testing
- Virtual Environments
- Package Management

**Expected Expert Concepts**:
- Metaclasses
- Async/Await Programming
- Multithreading and Multiprocessing
- Memory Management
- Performance Optimization
- Design Patterns
- Type Hints and mypy
- Advanced Testing (mocking, fixtures)
- Packaging and Distribution
- C Extensions and Cython

#### Build Progress Tracking System
- **Task**: Create a `progress_tracker.py` class for monitoring learning progress
- **Expected Output**: A progress tracking system with the following functionality:

**Key Methods**:
- `track_concept_study(concept, time_spent, score)` - Record study session
- `get_progress_summary()` - Show overall learning progress
- `calculate_mastery_level()` - Determine current skill level
- `get_recommended_next_concepts()` - Suggest what to study next
- `award_achievement(achievement_name)` - Give achievement badges
- `get_study_streak()` - Track consecutive study days

**Sample Implementation Structure**:
```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

class ProgressTracker:
    """Tracks learning progress and achievements."""
    
    def __init__(self, user_name: str = "Student"):
        self.user_name = user_name
        self.progress_file = Path(f"{user_name.lower()}_progress.json")
        self.progress_data: Dict = self._load_progress()
        
    def track_concept_study(self, concept: LearningConcept, 
                          time_spent: int, mastery_score: float) -> None:
        """Record a study session for a concept."""
        # Implementation for tracking study progress
        pass
        
    def get_progress_summary(self) -> Dict:
        """Get overall learning progress summary."""
        # Implementation for progress summary
        pass
        
    def calculate_current_level(self) -> SkillLevel:
        """Calculate current skill level based on mastery."""
        # Implementation for level calculation
        pass
        
    def get_achievements(self) -> List[str]:
        """Get list of earned achievements."""
        # Implementation for achievement system
        pass
```

#### Build the Interactive Console Application
- **Task**: Ask Copilot to create the main `main.py` with an engaging study interface
- **Expected Output**: A console application with the following menu structure:

**Sample Copilot Command**:
> "Create a main.py file that provides an interactive study session interface for the Python Study Buddy app. Include options to: 1) Start a personalized study session, 2) Review concept explanations, 3) Take a skill assessment, 4) View progress and achievements, 5) Practice coding challenges, 6) Set study goals. Include encouraging messages, progress bars, and celebration animations when milestones are achieved."

**Expected Console Output**:
```
🐍 Welcome to Python Study Buddy! 🐍
===================================
Hello, CodeLearner! Ready to level up your Python skills?

Current Level: Intermediate 🚀
Study Streak: 5 days 🔥
Concepts Mastered: 23/50 📚

🎯 Study Options:
1. Start Today's Study Session (Recommended: Decorators)
2. Browse Learning Concepts by Level
3. Take a Quick Skill Assessment
4. Practice Coding Challenges
5. View Progress Dashboard
6. Set Weekly Study Goals
7. Review Achievements & Badges
8. Exit

Choose your adventure: 1

🎯 Starting Study Session: Decorators
=====================================
⏱️  Estimated Time: 30 minutes
🎚️  Difficulty: 7/10
📈 Progress: This will advance you toward Advanced level!

Ready to begin? (y/n): y

🎉 Let's dive in! Remember: Every expert was once a beginner! 🎉
```

**Key Features**:
- Personalized greetings and progress updates
- Adaptive difficulty based on current skill level
- Gamification with streaks, badges, and celebrations
- Interactive coding challenges with immediate feedback
- Progress visualization and goal setting
- Motivational messaging and encouragement

### 2.4 GitHub Integration for Study Tracking

#### Create Study Milestone Issues
- **Task**: Use Copilot to create GitHub issues for study milestones
- Create issues for achieving specific learning goals
- Track concept mastery milestones
- Document learning journey and reflections

**Sample Copilot Commands**:
> "Create a GitHub issue celebrating completion of all beginner Python concepts"
> "Create an issue for setting up a study plan for intermediate OOP concepts"
> "Create an issue to track weekly coding challenge completions"

#### Learning Progress Repository
- **Task**: Use the repository to track your learning journey
- Commit code examples from study sessions
- Create branch for each major concept learned
- Document insights and "aha moments" in README

#### Verify Learning Progress
- Navigate to your repository
- Review the automatically created study milestone issues
- Examine your learning progress documentation
- Understand how MCP can enhance educational workflows

## Learning Outcomes
- How to configure and use MCP servers for educational applications
- Building interactive learning experiences with Python
- Progress tracking and gamification in educational software
- Creating adaptive content based on skill level
- GitHub integration for learning journey documentation
- Understanding how AI can personalize education experiences

---

> **Next Step**: Now that you've built a learning application with MCP, let's create an advanced MCP server for AI research discovery!

**Continue to**: [Part 3 - Building Your AI Research Learning Hub →](part3-ai-research-server.md)
