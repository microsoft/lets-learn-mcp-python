import asyncio
from enum import Enum
from typing import Any
from typing import List, Dict, Optional

from console_utils import (
    display_header,
    display_info_panel,
    display_success_panel,
    display_error_panel,
    display_code_panel,
    display_step,
    prompt_continue,
    section_separator,
    lab_complete,
    async_show_progress,
    console
)


# =============================================================================
# DEMO FUNCTIONS: Educational demonstrations of MCP concepts
# =============================================================================

def demo_tools():
    """Demonstrate how tools work in MCP."""
    display_header("Concept 1: Tools 🛠️")
    
    display_step(1, "What are Tools?", "Functions that the LLM can call to perform specific actions -basically function calling")
    
    content = """
    **Tools are executable functions** that extend what an LLM can do.
    
    Examples:
    - Send email
    - Update todos 
    - Run tests
    - File issues
    """
    display_info_panel(content, "🔧 Tools Explained")
    
    display_step(2, "Tool Definition", "How we define a tool in MCP")
    
    code_example = '''
import json
from dataclasses import dataclass
from typing import Dict, List
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent

# Initialize FastMCP server
mcp = FastMCP("Tools Demo Server")

@dataclass
class Exercise:
    title: str
    description: str
    hint: str
    solution: str
    difficulty: int

# Store exercises
exercises_db: Dict[str, List[Exercise]] = {}

@mcp.prompt()
async def generate_exercises(topic: str, level: str = "beginner") -> str:
    """Generate Python exercises prompt for a given topic and level."""
    
    return f"""Generate 5 Python exercises on '{topic}' for {level} level.

    Return ONLY valid JSON (no markdown, no extra text):
    {{
        "{level}": [
            {{
                "title": "Exercise Name",
                "description": "What to do",
                "hint": "Helpful hint",
                "solution": "Complete code solution",
                "difficulty": 1
            }}
        ]
    }}

    Make exercises progressively harder (difficulty 1-5)."""

@mcp.tool()
async def generate_and_create_exercises(
    topic: str, 
    level: str = "beginner",
    ctx: Context = None
) -> str:
    """Generate exercises using sampling and create them automatically."""
    
    try:
        # Get the prompt text
        prompt_text = await generate_exercises(topic, level)
        
        response = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt_text),
                )
            ],
            max_tokens=2000,
            )
        
        # Extract the text from the response
        response_text = response.content.text if response.content else ""
        
        # Parse the generated JSON
        exercises_data = json.loads(response_text)
        
        # Store exercises
        exercises_db[level] = []
        for ex in exercises_data[level]:
            exercises_db[level].append(Exercise(
                title=ex['title'],
                description=ex['description'],
                hint=ex['hint'],
                solution=ex['solution'],
                difficulty=ex['difficulty']
            ))
        
        return f"✅ Created {len(exercises_db[level])} exercises on '{topic}' for {level} level"
    
    except json.JSONDecodeError as e:
        return f"❌ JSON Error: {str(e)}\nResponse was: {response_text[:200]}..."
    except Exception as e:
        return f"❌ Error: {str(e)}"

@mcp.tool()
async def list_exercises() -> str:
    """List all created exercises."""
    
    if not exercises_db:
        return "No exercises yet. Use generate_and_create_exercises first!"
    
    result = []
    for level, exercises in exercises_db.items():
        result.append(f"\n{level.upper()} LEVEL:")
        for i, ex in enumerate(exercises):
            result.append(f"\n{i+1}. {ex.title}")
            result.append(f"   📝 {ex.description}")
            result.append(f"   💡 Hint: {ex.hint}")
            result.append(f"   ⭐ Difficulty: {ex.difficulty}/5")
    
    return "\n".join(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run())
        '''
    
    display_code_panel(code_example, "python", "📝 Tool Definition")
    
    display_step(3, "User Controls", "How to interact with tools")
    
    flow = """
    1. **Per-chat tool selection**: "Manually add or remove tools by clicking the icon in chat."
    2. **Mention tools in prompts**: "Use the # key to reference a tool in your prompt as needed."
    3. **User-defined tool sets**: "Re-use task specific tools across tasks. (use F1 or cmd+Shift+P 
    and search for 'configure toolsets' to set up your tool set)"
    """
    
    display_info_panel(flow, "🔄 Tool User Controls")

def demo_resources():
    """Demonstrate how resources work in MCP."""
    display_header("Concept 2: Resources")
    
    display_step(1, "What are Resources?", "File-like data that can be read")
    
    content = """
    **Resources are data sources** that the LLM can read from. They are read-only.
    They are identified using URI's that follow the following format:

    **[protocol]://[host]/[path]**
    
    eg. file:///home/user/documents/report.pdf or
    postgres://database/customers/schema
    
    Examples:
    - Files
    - Documents
    - Database entries/schemas
    - Images
    """
    display_info_panel(content, "Resources Explained")
    
    display_step(2, "Resource Definition", "How we define a resource in MCP")
    
    code_example = '''
    from pathlib import Path
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import SamplingMessage, TextContent
    from mcp import types
    import json
    import os

    # Create the MCP server
    mcp = FastMCP("Resources Demo Server")

    # File paths to the JSON files
    study_progress_file = os.path.join(os.path.dirname(__file__), "study_progress.json")
    beginner_exercises_file = os.path.join(os.path.dirname(__file__), "beginner_exercises.json")

    # ================================================================================
    # RESOURCES: File-like data that can be read by clients
    # =============================================================================

    @mcp.resource("user://study-progress/{username}")
    async def get_study_progress(username: str) -> str:
        """Get study progress for a user."""
        try:
            # Read study progress from JSON file
            with open(study_progress_file, 'r') as file:
                study_progress = json.load(file)
            
            # Check if the username matches (for this simple example)
            if study_progress.get("user_name") == username:
                return json.dumps(study_progress, indent=2)
            else:
                return json.dumps({
                    "error": f"No study progress found for user '{username}'"
                })
        except FileNotFoundError:
            return json.dumps({
                "error": "Study progress file not found"
            })
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid study progress file format"
            })

    # Add a resource to list all exercises
    @mcp.resource("user://exercises/{level}")
    async def list_exercises_for_level(level: str) -> str:
        """List all available exercises for a specific level."""
        try:
            # Only beginner exercises are available in the current implementation
            if level != "beginner":
                return json.dumps({
                    "error": f"No exercises found for level '{level}'"
                })
                
            # Read exercises from JSON file
            with open(beginner_exercises_file, 'r') as file:
                exercises = json.load(file)
                
            return json.dumps(exercises, indent=2)
        except FileNotFoundError:
            return json.dumps({
                "error": "Exercises file not found"
            })
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid exercises file format"
            })
        
    @mcp.tool()
    async def get_users_progress(
            username: str,
            ctx: Context = None
        ) -> str:
            """Get the study progress for a user."""

            try:
                # Get the prompt text
                user_progress_json = await get_study_progress(username)
                # Parse the generated JSON
                user_progress = json.loads(user_progress_json)
                prompt_text = f"""Here is the study progress for user '{username}':\n\n{json.dumps(user_progress, indent=2)}. 
                Return it to the user and suggest some topics they can study next based on their progress."""

                response = await ctx.session.create_message(
                    messages=[
                        SamplingMessage(
                            role="user",
                            content=TextContent(type="text", text=prompt_text),
                        )
                    ],
                    max_tokens=2000,
                    )
                
                # Extract the text from the response
                response_text = response.content.text if response.content else ""
                return response_text
            
            except Exception as e:
                return f"❌ Error: {str(e)}"


    if __name__ == "__main__":
        mcp.run()
    '''
        
    display_code_panel(code_example, "python", "📝 Resource Definition")
    
    display_step(3, "Resources Use Cases", "How LLMs can interact with resources")
    
    flow = """
    1. **Reduce response tokens**: "Returning embedded resources lets the agent pull data with less tokens."
    2. **Expose data/files to user**: "Provide assets for the user to act on, not just for LLM context e.g giving the user an image to see.
    3. **Attach as context**: "This reduces tool lookups with resources attached."
    """
    
    display_info_panel(flow, "Resource Use Cases")

def demo_prompts():
    """Demonstrate how prompts work in MCP."""
    display_header("Concept 3: Prompts")
    
    display_step(1, "What are Prompts?", "Pre-defined templates for specific tasks")
    
    content = """
    **Prompts are conversation templates** that help users accomplish tasks.
    They are explicitly invoked by the user unless otherwise specified.
    
    Examples:
    - Static presets 
    - Reusable placeholders
    - Dynamically generated
    """
    display_info_panel(content, "📝 Prompts Explained")
    
    display_step(2, "Prompt Definition", "How we define a prompt in MCP")
    
    code_example = '''
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Prompts Demo Server")

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

if __name__ == "__main__":
    mcp.run()"""
    '''
    
    display_code_panel(code_example, "python", "📝 Prompt Definition")
    
    display_step(3, "Prompts Use Cases", "How can prompts be useful?")
    
    flow = """
    1. **Onboarding Prompts**: "Welcome prompts for users to verify setup and tour tools."
    2. **Common workflows**: "One-shot tasks, parameterized for reusability."
    3. **Context-aware workflows**: "Dynamically customized for the current user/codebase."
    """
    
    display_info_panel(flow, "Prompts Use Cases")

def demo_running_server():
    """Show how to run the MCP server."""
    display_header("Running Your MCP Server 💻✨")
    
    display_step(1, "Server Setup", "Creating the server in Python")
    
    code_example = '''from mcp.server.fastmcp import FastMCP

    # Create the MCP server
    mcp = FastMCP("mcp-concepts-demo")

    # Add your tools, resources, and prompts here...

    if __name__ == "__main__":
        # Run the server
        mcp.run()'''
    
    display_code_panel(code_example, "python", "Server Setup")

    display_step(2, "Running your MCP Server in VS Code", "How to configure your MCP server")
    
    flow = """
    1. Create a `.vscode` folder in your project directory
    2. Create an `mcp.json` file inside `.vscode`
    3. Add the following configuration:

    To get the command for UV run `which uv` in your terminal.
    For the command you can also use `python path-to-your-server-file`
    """
    display_info_panel(flow, "MCP Configuration")

    code_example = '''
    {
        "inputs": [
        ],
        "servers": {
            "learnpython-mcp": {
                "command": "/opt/anaconda3/bin/uv",
                "args": [
                    "--directory",
                    ".",
                    "run",
                    "prompts_server.py"
                ],
            }
        }
    }
    '''
    display_code_panel(code_example, "json", "📝 MCP VS Code Configuration")


async def main():
    """Main demo function - step by step through MCP concepts."""
    try:
        display_header("🎓 Lab 2: Understanding MCP Core Concepts")
        
        display_info_panel(
            "This lab teaches the **three core concepts** of MCP:\n\n"
            "• **Prompts** - Pre-written templates for tasks\n\n"
            "• **Tools** - Functions that can be executed\n\n"
            "• **Resources** - File-like data that can be read\n\n"
            "• **Bonus: Sampling** - Letting the client perform an LLM call\n\n"
            "We'll go step by step through each concept!",
            "📚 Learning Objectives"
        )

        # Step 1: Creating and configuring the server
        prompt_continue("Press Enter to learn how to create your server...")
        section_separator()
        demo_running_server()

        # Step 2: Prompts
        prompt_continue("Press Enter to learn about Prompts...")
        section_separator()
        demo_prompts()

        # Step 3: Tools
        prompt_continue("Press Enter to learn about Tools...")
        section_separator()
        demo_tools()

        # Step 4: Resources
        prompt_continue("Press Enter to learn about Resources...")
        section_separator()
        demo_resources()
        
        
        # Completion
        section_separator()
        lab_complete()
        
        
        display_info_panel(
            "**Next Steps:**\n\n"
            "• Try running this server: `server_part2.py`\n"
            "• Test it with VS Code to get the full workflow going\n"
            "• Experiment with adding your own tools and resources!\n"
            "• Check out part3 for building more advanced servers"
        )
        
    except KeyboardInterrupt:
        console.print("\n\n👋 Demo interrupted. Come back anytime!", style="yellow")
    except Exception as e:
        display_error_panel(f"Demo error: {str(e)}", "❌ Error")

if __name__ == "__main__":
    # Run the educational demo
    asyncio.run(main())
