# AutoGen Project Description for LLM

## Project Overview
AutoGen is a framework for building and orchestrating Language Model (LLM) based AI agents. The project enables the development of conversational AI systems where multiple agents can interact, collaborate, and perform complex tasks.

## Core Components and Structure

### Main Packages (`/python/packages/`)
- `autogen-core`: Core functionality and base components
- `autogen-agentchat`: Agent conversation and interaction framework
- `autogen-ext`: Extensions and additional integrations
- `autogen-studio`: Web-based UI for prototyping and managing AI agents

### Project Organization
- Source code: `/python/packages/`
- Documentation: `/docs/`
- Protocol definitions: `/protos/`
- Templates: `/templates/`
- Examples: `/python/samples/`

## Development Context
- Primary language: Python
- Package management: Uses pyproject.toml for dependencies

## Key Concepts
1. Agent System
   - Multi-agent conversations
   - Agent roles and capabilities
   - Message passing and interaction protocols

2. Core Components
   - Agent context management
   - Event handling
   - Service integration

3. Extension System
   - Custom agent behaviors
   - External service integration
   - Additional capabilities

4. AutoGen Studio (UI)
   - Web-based interface for agent prototyping
   - Components:
     - Frontend: React-based web interface
     - Backend: FastAPI server
     - Database: SQLModel (Pydantic + SQLAlchemy)
   - Features:
     - Agent workflow creation and management
     - Skill enhancement and composition
     - Interactive agent testing
     - Multiple database backend support (SQLite, PostgreSQL, MySQL, etc.)

## Code Modification Guidelines
1. Agent Changes:
   - Maintain separation between core, chat, and extension functionalities
   - Follow existing agent interaction patterns
   - Preserve message handling protocols

2. System Integration:
   - Use established service integration patterns
   - Follow type hinting and documentation standards

3. Extension Development:
   - Keep extensions modular and self-contained
   - Document dependencies and requirements
   - Include appropriate test coverage

4. UI Development (AutoGen Studio):
   - Follow React component patterns in frontend
   - Maintain API consistency in FastAPI backend
   - Use SQLModel for database operations
   - Support multiple database backends

## Important Files
- `pyproject.toml`: Project dependencies and build configuration
- `README.md`: Main documentation and getting started guide
- `.github/`: CI/CD workflows and GitHub configurations
- `docs/`: Comprehensive documentation and guides