# LETSLEARNMCP-PYTHON

# Let's Learn MCP with Python - Tutorial Series

A comprehensive guide to understanding and building Model Context Protocol (MCP) Servers for Python developers.

## What You'll Build

By the end of this tutorial series, you'll have:

1. **🎮 MyGame App** - A fun console application that uses existing MCP servers to manage classic Nintendo game data and create GitHub issues
2. **🍽️ Lunch Roulette MCP Server** - Your own custom MCP server that helps AI assistants pick restaurants and track dining preferences

## Tutorial Structure

### [Part 1: Prerequisites and Setup](part1-setup-python.md)
**⏱️ Time: 15-20 minutes**

Set up your development environment and understand MCP fundamentals:
- Install VS Code, Python 3.12+, and Python extension
- Learn what Model Context Protocol is and why it matters
- Understand the client-server architecture
- Verify your development environment

**Start here**: [Part 1: Prerequisites and Setup →](part1-setup-python.md)

---

### [Part 2: Using MCP Servers - MyGame App](part2-game-app-python.md)
**⏱️ Time: 15-30 minutes**

Learn to use existing MCP servers by building a Nintendo games-themed application:
- Configure GameMCP and GitHub MCP servers
- Create Python models for game data using dataclasses
- Build an interactive console application with ASCII art
- Integrate with GitHub to create issues programmatically
- Understand how AI assistants interact with MCP servers

**What you'll create**:
```
🎮 Random Game Picker!
======================
Name: Super Mario Bros.
Platform: NES
Year: 1985
Genre: Platformer
Details: A legendary side-scrolling platformer...
```

**Continue to**: [Part 2: MyGame App →](part2-game-app-python.md)

---

### [Part 3: Building Your Own MCP Server - Lunch Roulette](part3-lunch-server-python.md)
**⏱️ Time: 30-45 minutes**

Build a complete MCP server from scratch:
- Create a restaurant management service with JSON persistence
- Implement four MCP tools for AI assistant integration
- Handle async operations and error handling
- Test and deploy your server
- Understand MCP protocol implementation details

**What you'll build**:
- `get_restaurants()` - List all available restaurants
- `add_restaurant()` - Add new dining options
- `pick_random_restaurant()` - AI-powered lunch selection
- `get_visit_statistics()` - Track dining patterns

**Continue to**: [Part 3: Lunch Roulette Server →](part3-lunch-server-python.md)

---

## Quick Start

If you're ready to dive in immediately:

1. **Prerequisites**: Ensure you have VS Code, Python 3.12+, and Python extension installed
2. **Choose your path**:
   - 🆕 **New to MCP?** Start with [Part 1: Setup](part1-setup-python.md)
   - 🔧 **Want to use MCP?** Jump to [Part 2: MyGame App](part2-game-app-python.md)
   - 🏗️ **Ready to build?** Go to [Part 3: Lunch Server](part3-lunch-server-python.md)

## Learning Outcomes

After completing this tutorial series, you'll understand:

- ✅ **MCP Fundamentals** - What MCP is and how it enables AI-tool integration
- ✅ **Consumer Patterns** - How to use existing MCP servers in your applications
- ✅ **Server Development** - How to build and deploy your own MCP servers
- ✅ **Python Best Practices** - Async programming, dataclasses, and JSON handling
- ✅ **AI Integration** - How AI assistants interact with your tools and services

## Technology Stack

This tutorial uses modern Python development practices:

- **Python 3.12+** - Latest Python runtime with performance improvements
- **asyncio** - Asynchronous programming for MCP servers
- **dataclasses** - Modern Python data modeling
- **json** - Built-in JSON serialization
- **pathlib** - Modern file path handling
- **mcp** - Official Python SDK for MCP server development

## Repository Structure

```
letslearnmcp-python/
├── README-python.md             # This overview
├── part1-setup-python.md        # Prerequisites and environment setup
├── part2-game-app-python.md     # Using existing MCP servers
└── part3-lunch-server-python.md # Building your own MCP server
```

## Additional Resources

- 📖 [MCP Official Documentation](https://modelcontextprotocol.io/)
- 🛠️ [Python MCP SDK Repository](https://github.com/modelcontextprotocol/python-sdk)
- 🐍 [Python MCP Examples](https://github.com/modelcontextprotocol/servers)
- 🎮 [MyGame App Python Version](https://github.com/jamesmontemagno/MyGameAppMCP-Python)
- 🍽️ [Lunch Roulette Python Version](https://github.com/jamesmontemagno/LunchRouletteMCP-Python)

## Contributing

This tutorial is open source! Feel free to:
- 🐛 Submit improvements and corrections
- 💡 Add more examples and use cases
- 🤝 Share your own MCP server implementations
- 💬 Help others in the discussions

---

**Ready to get started?** Begin with [Part 1: Prerequisites and Setup →](part1-setup-python.md)

*Happy coding! 🎮🍽️*

