"""
Lab 2: Understanding MCP Core Concepts
This lab demonstrates the three core concepts of MCP: Prompts, Tools, and Resources.
"""

from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# Create the MCP server
mcp = FastMCP("Python Study Buddy")

# =============================================================================
# PROMPTS: Pre-written templates that help users accomplish specific tasks
# =============================================================================

@mcp.prompt()
def python_topics(level: str = "beginner") -> str:
    """List Python topics based on user experience level."""

    level = level.lower()

    learning_levels = {
        "beginner": "for someone new to programming",
        "intermediate": "for someone with some intermediate programming experience",
        "advanced": "for someone with extensive programming experience",
    }

    prompt = f"generate 5 Python topics {learning_levels[level]}, numbered from most fundamental to most complex. After listing the topics, ask if they'd like to try exercises for any topic (recommend starting with #1)."

    # Return a more direct prompt that's easier for the LLM to follow
    return prompt

# =============================================================================
# TOOLS: Functions that can be called by the LLM (with user approval)
# =============================================================================

@dataclass
class Exercise:
    title: str
    description: str
    hint: str
    solution: str
    difficulty: int

# Store exercises in a persistent way
# In production, you'd want to use a database or file storage
exercises_db: Dict[str, List[Exercise]] = {
    "beginner": [Exercise(
        title="Hello World",
        description="Write a program that prints 'Hello, Python!' to the console",
        hint="Try using print()",
        solution="print('Hello, World!')",
        difficulty=1
    )],
    "intermediate": [Exercise(
        title="List Comprehension",
        description="Create a list of squares for numbers 1-10 using list comprehension",
        hint="Use the syntax: [expression for item in range()]",
        solution="squares = [x**2 for x in range(1, 11)]\nprint(squares)",
        difficulty=3
    )],
    "advanced": [Exercise(
        title="Decorator Pattern",
        description="Create a decorator that times how long a function takes to run",
        hint="Use time.time() before and after the function call",
        solution="import time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end-start:.4f} seconds')\n        return result\n    return wrapper",
        difficulty=5
    )]
}

@mcp.prompt()
def generate_exercises(
    topic: str,
    level: str = "beginner"
) -> str:
    """Generate Python exercises based on a topic and user experience level."""

    prompt = f"""You are a Python study buddy assistant. The user wants to practice 5 exercises on the topic '{topic}' at the {level} level.

    Please do the following:
    1. Generate 5 Python exercises related to '{topic}' for a {level} level learner.
    2. Each exercise should include:
       - Title
       - Description
       - Hint
       - Solution
       - Difficulty (1-5)
    3. Return the exercises as a JSON string that is a dictionary with the level as the key 
    and a list of exercises as the value. Make sure the format is compatible with the `create_exercise` tool.
    Do not include "exercises_data" as a key. Just use the level as the highest key. and include the data in that. 
    """
    return prompt

@mcp.tool()
def create_exercise(
    exercises_data: Dict,
    level: str = "beginner"
) -> str:
    """Use this tool to create new Python exercises from a dictionary containing generated exercises."""
    
    # Clear existing exercises for this level and create new ones
    exercises_db[level] = []
    
    exercises_list = exercises_data[level]

    for exercise_data in exercises_list:
        exercise = Exercise(
            title=exercise_data['title'],
            description=exercise_data['description'],
            hint=exercise_data['hint'],
            solution=exercise_data['solution'],
            difficulty=exercise_data['difficulty']
        )
        exercises_db[level].append(exercise)

    return json.dumps({
        "success": True,
        "message": f"Successfully created {len(exercises_list)} exercises for level '{level}'.",
        "exercises_count": len(exercises_list)
    })

@dataclass
class StudyProgress:
    """Tracks learning progress"""
    user_name: str
    level: str
    completed_exercises: List[str]
    current_streak: int
    total_exercises_attempted: int
    start_date: str
    achievements: List[str]

study_progress_db: Dict[str, StudyProgress] = {
    "python_user_1": StudyProgress(
        user_name="python_user_1",
        level="beginner",
        completed_exercises=["Hello World"],
        current_streak=1,
        total_exercises_attempted=1,
        start_date="2023-01-01",
        achievements=["First Exercise Completed"]
    )
}


@mcp.tool()
def track_study_progress(
    user_name: str = "user_1",
    level: str = "beginner",
    completed_exercises: List[str]=[],
    current_streak: int = 0,
    total_exercises_attempted: int = 0,
    start_date: str = datetime.now().isoformat(),
    achievements: List[str] = []
) -> str:
    """Track the study progress of a user."""

    progress = StudyProgress(
        user_name=user_name,
        level=level,
        completed_exercises=completed_exercises,
        current_streak=current_streak,
        total_exercises_attempted=total_exercises_attempted,
        start_date=start_date,
        achievements=achievements
    )

    # Here you would typically save the progress to a database
    return json.dumps({
        "success": True,
        "message": f"Study progress for {user_name} tracked successfully",
        "progress": asdict(progress)
    })

@mcp.tool()
def start_study_buddy(username: str, level: str = "beginner", run_in_terminal: bool = False) -> str:
    """Start a study buddy session for the user using the current exercises_db and progress."""
    from rich.console import Console
    import os
    import json as json_lib
    
    console = Console()
    
    try:
        # Save exercises to file system for run_study_buddy.py to access
        if level in exercises_db and exercises_db[level]:
            # Create the exercises JSON file
            exercise_file_path = os.path.join(os.getcwd(), f'{level}_exercises.json')
            with open(exercise_file_path, 'w') as f:
                exercises_json = [
                    {
                        'title': ex.title,
                        'description': ex.description,
                        'hint': ex.hint,
                        'solution': ex.solution,
                        'difficulty': ex.difficulty
                    }
                    for ex in exercises_db[level]
                ]
                json_lib.dump(exercises_json, f, indent=2)
                console.print(f"[green]Created exercise file: {exercise_file_path}[/green]")

        # Get the absolute path to the run_study_buddy.py script
        script_path = os.path.join(os.getcwd(), 'run_study_buddy.py')
        
        # Pass the username and level as command line arguments
        terminal_cmd = f"python {script_path} {username} {level}"
        
        console.print(f"\n[bold blue]🚀 Study Buddy prepared for {username}![/bold blue]")
        
        # If run_in_terminal is True, directly run the command using run_in_terminal_cmd
        if run_in_terminal:
            return run_in_terminal_cmd(terminal_cmd)
        else:
            return json.dumps({
                "success": True,
                "message": f"Study Buddy prepared for {username} at {level} level. Run with command: {terminal_cmd}",
                "terminal_command": terminal_cmd
            })
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]👋 Study session interrupted. Come back anytime![/yellow]")
        return json.dumps({
            "success": False,
            "message": "Study session was interrupted by user"
        })
    except Exception as e:
        console.print(f"\n[red]❌ An error occurred: {e}[/red]")
        return json.dumps({
            "success": False,
            "message": f"An error occurred: {str(e)}"
        })


@mcp.tool()
def run_in_terminal_cmd(command: str) -> str:
    """Run a command in the terminal."""
    import subprocess
    import sys
    import os
    from rich.console import Console
    
    console = Console()
    
    try:
        # Check if this is a study buddy command
        is_study_buddy = "run_study_buddy.py" in command
        
        if is_study_buddy:
            console.print("[bold green]🚀 Starting Python Study Buddy...[/bold green]")
            
            # For Study Buddy, we want to use a direct approach that works in the current terminal
            # This will work better with VS Code's integrated terminal
            
            # Get the absolute path for the command
            if command.startswith("python "):
                script_path = command.split(" ")[1]
                if not os.path.isabs(script_path):
                    script_path = os.path.join(os.getcwd(), script_path)
                
                # Extract arguments
                args = command.split(" ")[2:]
                
                # Run in current process
                result = subprocess.run(
                    ["python", script_path] + args,
                    text=True,
                    capture_output=False
                )
                
                return json.dumps({
                    "success": result.returncode == 0,
                    "message": "Python Study Buddy session has completed!"
                })
        else:
            # For non-study buddy commands, use the previous approach
            if sys.platform == 'darwin':  # macOS
                subprocess.Popen(['osascript', '-e', f'tell application "Terminal" to do script "{command}"'])
            elif sys.platform == 'win32':  # Windows
                subprocess.Popen(['cmd.exe', '/c', command], shell=True)
            else:  # Linux and other Unix-like
                subprocess.Popen(['xterm', '-e', command])
            
            return json.dumps({
                "success": True,
                "message": f"Command '{command}' is being executed in a terminal window."
            })
    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"Failed to execute command in terminal: {str(e)}"
        })


# ================================================================================
# RESOURCES: File-like data that can be read by clients
# =============================================================================

@mcp.resource("user://exercises/{level}")
def get_exercises(level: str) -> str:
    """Get exercises by level of experience."""
    exercises = exercises_db.get(level, [])
    if exercises:
        return json.dumps([asdict(ex) for ex in exercises], indent=2)
    else:       
        return json.dumps({
            "error": f"No exercises found for level '{level}'"
        }) 
    

# Add a resource to list all exercises
@mcp.resource("user://exercises")
def list_exercises() -> str:
    """List all available exercises."""

    all_exercises = []

    for level in exercises_db.keys():
        all_exercises.append(exercises_db[level])
    return json.dumps({
        "exercises": all_exercises
    }, indent=2)

if __name__ == "__main__":
    mcp.run()
    # start_study_buddy("marlene", "beginner")

