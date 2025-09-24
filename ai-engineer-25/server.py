from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent

# Create the MCP server
mcp = FastMCP()

# =============================================================================
# TOOLS: Functions that can be called by the LLM
# =============================================================================

@mcp.tool(name="get_speaker_session", 
          description="Get information about a specific speaker's sessions at the AI Engineer Conference")
async def get_speaker_session(speaker_name: str, ctx: Context = None) -> str:

    with open('schedule.txt') as f:
        schedule = f.read()
    
    prompt = f"""Based on the following conference schedule, 
            tell me about the sessions for the speaker named {speaker_name}: {schedule}"""
    
    response = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            )
        ],
        max_tokens=1000
    )
    
    if response.content.type == "text":
        return response.content.text
    else:
        return str(response.content)


# =============================================================================
# PROMPTS: Pre-written templates that help users accomplish specific tasks
# =============================================================================

# @mcp.prompt(name="get_speaker_session_prompt", 
#             description="Return a prompt to get information about a speaker's sessions at the AI Engineer Conference")
# def get_speaker_session_prompt(speaker_name: str) -> str:
    
#     prompt = f"""Which sessions is {speaker_name} giving at the AI Engineer Conference?
#     If the speaker works at Github, start by saying '✨✨✨This is one of my creators! ✨✨✨'. 
#     Make sure to use the tool 'get_speaker_session' to get the information.
#     """

    # Return a more direct prompt that's easier for for the LLM to follow
    return prompt

# ================================================================================
# RESOURCES: File-like data that can be read by clients
# =============================================================================

# @mcp.resource("file://documents/{name}")
# def read_document(name: str) -> str:
#     """Read a document by name."""
#     # This would normally read from disk
#     with open(f"{name}") as f:
#         return f.read()

if __name__ == "__main__":
    mcp.run()