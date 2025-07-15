"""
Simple Study Buddy - A simplified Python learning application
"""

from dataclasses import dataclass
from typing import Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.markdown import Markdown
import json

@dataclass
class Exercise:
    """Represents a single exercise"""
    title: str
    description: str
    hint: str
    solution: str
    difficulty: int

@dataclass
class StudyProgress:
    """Tracks user's learning progress"""
    user_name: str
    level: str
    completed_exercises: List[str]
    current_streak: int
    total_exercises_attempted: int
    start_date: str
    achievements: List[str]

class PythonStudyBuddy:
    """Main study buddy application class"""
    
    def __init__(self, custom_exercises: Dict[str, List[Exercise]] = None, custom_progress: StudyProgress = None):
        self.console = Console()
        self.exercises = custom_exercises or {}
        self.progress = custom_progress or StudyProgress(
            user_name="Student",
            level="beginner",
            completed_exercises=[],
            current_streak=0,
            total_exercises_attempted=0,
            start_date="",
            achievements=[]
        )
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = f"""
# 🐍 Python Study Buddy

Welcome, **{self.progress.user_name}**! 
Ready to practice Python at the **{self.progress.level}** level?

Current Streak: 🔥 {self.progress.current_streak} days
Exercises Completed: ✅ {len(self.progress.completed_exercises)}
        """
        self.console.print(Panel(Markdown(welcome_text), border_style="blue"))
    
    def display_exercise(self, exercise: Exercise):
        """Display a single exercise"""
        self.console.print(f"\n[bold cyan]📝 {exercise.title}[/bold cyan]")
        self.console.print(f"[yellow]Difficulty: {'⭐' * exercise.difficulty}[/yellow]\n")
        self.console.print(Panel(exercise.description, title="Description", border_style="green"))
    
    def get_exercise_choice(self) -> Exercise:
        """Let user choose an exercise"""
        exercises_list = self.exercises.get(self.progress.level, [])
        
        if not exercises_list:
            self.console.print("[red]No exercises available for your level![/red]")
            return None
        
        # Display available exercises
        table = Table(title=f"Available {self.progress.level.title()} Exercises")
        table.add_column("No.", style="cyan", width=4)
        table.add_column("Title", style="magenta")
        table.add_column("Difficulty", style="yellow")
        table.add_column("Status", style="green")
        
        for idx, exercise in enumerate(exercises_list, 1):
            status = "✅ Completed" if exercise.title in self.progress.completed_exercises else "📝 Not started"
            table.add_row(
                str(idx),
                exercise.title,
                "⭐" * exercise.difficulty,
                status
            )
        
        self.console.print(table)
        
        # Get user choice
        choice = IntPrompt.ask(
            "\n[bold]Choose an exercise (number)[/bold]",
            default=1,
            choices=[str(i) for i in range(1, len(exercises_list) + 1)]
        )
        
        return exercises_list[choice - 1]
    
    def practice_exercise(self, exercise: Exercise):
        """Practice a single exercise"""
        self.display_exercise(exercise)
        
        # Show hint option
        show_hint = Prompt.ask("\n[yellow]Need a hint? (y/n)[/yellow]", default="n")
        if show_hint.lower() == 'y':
            self.console.print(Panel(f"💡 Hint: {exercise.hint}", border_style="yellow"))
        
        # Get user's solution
        self.console.print("\n[bold]Write your solution:[/bold]")
        user_solution = Prompt.ask(">>> ")
        
        # Show solution
        show_solution = Prompt.ask("\n[yellow]Ready to see the solution? (y/n)[/yellow]", default="y")
        if show_solution.lower() == 'y':
            self.console.print(Panel(
                f"✨ Solution:\n{exercise.solution}",
                title="Solution",
                border_style="green"
            ))
        
        # Mark as completed
        completed = Prompt.ask("\n[bold]Did you complete this exercise? (y/n)[/bold]", default="y")
        if completed.lower() == 'y' and exercise.title not in self.progress.completed_exercises:
            self.progress.completed_exercises.append(exercise.title)
            self.progress.total_exercises_attempted += 1
            self.console.print("[green]✅ Great job! Exercise marked as completed![/green]")
    
    def run_study_session(self):
        """Run the main study session"""
        self.display_welcome()
        
        while True:
            exercise = self.get_exercise_choice()
            if not exercise:
                break
            
            self.practice_exercise(exercise)
            
            # Ask if user wants to continue
            continue_session = Prompt.ask(
                "\n[bold]Continue with another exercise? (y/n)[/bold]",
                default="y"
            )
            if continue_session.lower() != 'y':
                break
        
        # Show summary
        self.console.print(Panel(
            f"""
[bold green]Session Summary[/bold green]
Exercises Completed: {len(self.progress.completed_exercises)}
Total Attempts: {self.progress.total_exercises_attempted}

Keep up the great work! 🎉
            """,
            border_style="green"
        ))

