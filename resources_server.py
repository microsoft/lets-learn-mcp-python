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
