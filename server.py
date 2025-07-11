import subprocess

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("My App")


@mcp.prompt()
def python_topics(code: str, level: str = "beginner") -> str:
    """List out 5 Python topics a user should learn at different levels of detail"""
    levels = {
        "beginner": "Please list 5 Python topics for someone new to programming",
        "intermediate": "Please list 5 Python topics with moderate technical detail",
        "advanced": "Please list 5 Python topics with a detailed technical analysis",
    }
    explanation_request = levels[level]
    return f"{explanation_request}:\n\n```python\n{code}\n```"



@mcp.tool(
    description="Add a list of numbers and return structured result with metadata",
    annotations=ToolAnnotations(title="Number Addition Tool", idempotentHint=True, readOnlyHint=True),
)
def add_list_of_numbers(numbers: list[int]) -> dict:
    """Add a list of numbers and return structured result"""
    total = sum(numbers)
    return {
        "numbers": numbers,
        "sum": total,
        "count": len(numbers),
        "average": total / len(numbers) if numbers else 0,
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
        "summary": f"Sum of {numbers} = {total}",
    }


@mcp.tool(
    description="Count occurrences of a letter in text with detailed analysis",
    annotations=ToolAnnotations(title="Letter Counter Tool", idempotentHint=True, readOnlyHint=True),
)
def count_letter_in_text(text: str, letter: str) -> dict:
    """Count occurrences of a letter in a text with structured output"""
    count = text.count(letter)
    text_length = len(text)
    percentage = (count / text_length * 100) if text_length > 0 else 0

    return {
        "letter": letter,
        "count": count,
        "text_length": text_length,
        "percentage": round(percentage, 1),
        "summary": (
            f"Letter '{letter}' appears {count} times in the text " f"(length: {text_length} chars, {percentage:.1f}%)"
        ),
    }



if __name__ == "__main__":
    mcp.run()
