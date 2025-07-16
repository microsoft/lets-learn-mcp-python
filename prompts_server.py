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
    mcp.run()
