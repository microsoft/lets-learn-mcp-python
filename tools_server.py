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
    """Generate Python exercises for a given topic and level."""
    
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
