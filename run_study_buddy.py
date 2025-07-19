"""
Run the Python Study Buddy in a terminal
This script creates a study buddy instance based on command line arguments.
"""

import sys
import os
from datetime import datetime

def main():
    """Run the study buddy using command line arguments."""
    if len(sys.argv) < 3:
        print("Error: Missing required arguments.")
        print("Usage: python run_study_buddy.py <username> <level>")
        sys.exit(1)
    
    username = sys.argv[1]
    level = sys.argv[2]
    
    try:
        # Import the necessary modules
        from simple_study_buddy import PythonStudyBuddy, StudyProgress
        
        print(f"\n{'=' * 60}")
        print(f"🐍 Python Study Buddy - Terminal Session")
        print(f"{'=' * 60}")
        print(f"👤 Username: {username}")
        print(f"📊 Level: {level}")
        print(f"📅 Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'=' * 60}\n")
        
        # Load the exercise data based on the level
        exercise_data = load_exercises(level)
        
        if not exercise_data or level not in exercise_data or not exercise_data[level]:
            print(f"\n⚠️ Warning: No {level} exercises found. Falling back to default exercises.")
        
        # Create user progress
        user_progress = StudyProgress(
            user_name=username,
            level=level,
            completed_exercises=[],
            current_streak=0,
            total_exercises_attempted=0,
            start_date=datetime.now().isoformat(),
            achievements=[]
        )
        
        # Create the study buddy instance
        study_buddy = PythonStudyBuddy(
            custom_exercises=exercise_data,
            custom_progress=user_progress
        )
        
        # Print welcome message
        print(f"\n🚀 Starting Python Study Buddy session for {username} at {level} level!\n")
        
        # Run the study buddy
        study_buddy.run_study_session()
        
        print("\nSession completed. Thanks for studying Python!")
    
    except Exception as e:
        print(f"\n❌ Error: Failed to run study buddy: {str(e)}")
        sys.exit(1)

def load_exercises(level):
    """Load exercises for the given level."""
    import json
    import os
    from simple_study_buddy import Exercise
    
    exercises = {}
    exercise_files_found = False
    
    # First, look for specific level exercise file (e.g., beginner_exercises.json)
    try:
        level_file = f"{level}_exercises.json"
        if os.path.exists(level_file):
            print(f"📚 Loading exercises from {level_file}...")
            with open(level_file, 'r') as f:
                exercise_data = json.load(f)
                exercises[level] = []
                
                for ex in exercise_data:
                    exercises[level].append(Exercise(
                        title=ex['title'],
                        description=ex['description'],
                        hint=ex['hint'],
                        solution=ex['solution'],
                        difficulty=ex['difficulty']
                    ))
                print(f"✅ Loaded {len(exercises[level])} exercises for {level} level!")
                exercise_files_found = True
    except Exception as e:
        print(f"❌ Could not load {level}_exercises.json: {e}")
    
    # If specific level file not found, look for any exercise files
    if not exercise_files_found:
        try:
            # Look for exercise files in the current directory
            exercise_files = [f for f in os.listdir() if f.endswith('_exercises.json')]
            
            if exercise_files:
                print(f"📚 Found {len(exercise_files)} exercise files.")
                
            for file in exercise_files:
                with open(file, 'r') as f:
                    try:
                        exercise_data = json.load(f)
                        level_key = file.split('_')[0]  # Get the level from filename
                        if level_key not in exercises:
                            exercises[level_key] = []
                        
                        for ex in exercise_data:
                            exercises[level_key].append(Exercise(
                                title=ex['title'],
                                description=ex['description'],
                                hint=ex['hint'],
                                solution=ex['solution'],
                                difficulty=ex['difficulty']
                            ))
                        print(f"✅ Loaded {len(exercises[level_key])} exercises for {level_key} level!")
                        exercise_files_found = True
                    except:
                        print(f"❌ Could not parse {file}")
        except Exception as e:
            print(f"❌ Error loading exercises from files: {e}")
    
    # If no exercises found, provide fallback exercises
    if not exercise_files_found or level not in exercises or not exercises[level]:
        print("ℹ️ No appropriate exercises found. Creating default exercises...")
        return create_default_exercises()
    
    return exercises

def create_default_exercises():
    """Create default exercises as a fallback."""
    from simple_study_buddy import Exercise
    
    return {
        "beginner": [
            Exercise(
                title="Hello Variables",
                description="Create a variable called `name` and assign your name to it. Then print 'Hello, ' followed by your name.",
                hint="Use the print() function and string concatenation with '+' or f-strings to combine text.",
                solution='name = "Marlene"\nprint("Hello, " + name)\n# Or using f-strings:\n# print(f"Hello, {name}")',
                difficulty=1
            ),
            Exercise(
                title="Basic Data Types",
                description="Create four variables: an integer called `age` with value 25, a float called `height` with value 1.75, a boolean called `is_student` with value True, and a string called `favorite_color` with a color of your choice. Print each variable on a separate line with a description.",
                hint="Make sure to use the correct data type for each variable. Use print() with descriptive text.",
                solution='age = 25\nheight = 1.75\nis_student = True\nfavorite_color = "blue"\n\nprint("Age:", age)\nprint("Height:", height)\nprint("Is student?", is_student)\nprint("Favorite color:", favorite_color)',
                difficulty=2
            ),
            Exercise(
                title="Type Conversion",
                description='Create a variable `user_input` with the string value "42". Convert it to an integer, multiply it by 2, and print the result.',
                hint="Use the int() function to convert a string to an integer.",
                solution='user_input = "42"\nconverted_number = int(user_input)\nresult = converted_number * 2\nprint(result)',
                difficulty=2
            ),
            Exercise(
                title="Basic Math Operations",
                description="Create two variables: `x` with value 10 and `y` with value 3. Calculate and print the following operations: addition, subtraction, multiplication, division, integer division, modulus, and exponentiation of these two variables.",
                hint="Python uses standard math operators: +, -, *, /, //, %, and ** for exponentiation.",
                solution="x = 10\ny = 3\n\nprint(\"Addition:\", x + y)\nprint(\"Subtraction:\", x - y)\nprint(\"Multiplication:\", x * y)\nprint(\"Division:\", x / y)\nprint(\"Integer Division:\", x // y)\nprint(\"Modulus:\", x % y)\nprint(\"Exponentiation:\", x ** y)",
                difficulty=3
            ),
            Exercise(
                title="String Operations",
                description="Create a variable `first_name` with your first name and `last_name` with your last name. Combine them to create a `full_name` variable. Then print the full name, the length of your full name, and your full name in all uppercase letters.",
                hint="Use string concatenation to combine strings and the len() function to get string length. The upper() method converts a string to uppercase.",
                solution='first_name = "John"\nlast_name = "Doe"\nfull_name = first_name + " " + last_name\n\nprint("Full name:", full_name)\nprint("Length of full name:", len(full_name))\nprint("Uppercase full name:", full_name.upper())',
                difficulty=3
            )
        ],
        "intermediate": [
            Exercise(
                title="List Comprehension",
                description="Create a list of squares for numbers 1-10 using list comprehension",
                hint="Use the syntax: [expression for item in range()]",
                solution="squares = [x**2 for x in range(1, 11)]\nprint(squares)",
                difficulty=3
            )
        ],
        "advanced": [
            Exercise(
                title="Decorator Pattern",
                description="Create a decorator that times how long a function takes to run",
                hint="Use time.time() before and after the function call",
                solution="import time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end-start:.4f} seconds')\n        return result\n    return wrapper",
                difficulty=5
            )
        ]
    }

if __name__ == "__main__":
    main()
