from deepagents import create_deep_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver

from config import settings
from prompts import (
    CHUNK_ANALYST_INSTRUCTIONS,
    RAG_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from tools import backend, search_documents

MAIN_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
ANALYST_MODEL = "nvidia/nemotron-3.5-lightning:free"
MAX_CONCURRENT_ANALYSTS = 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=MAX_CONCURRENT_ANALYSTS,
    )
)


def build_agent():
    """
    Create the RAG agent: main model with the search tool and a chunk-analyst subagent.
    """
    chunk_analyst = {
        "name": "chunk-analyst",
        "description": (
            "Analyze one retrieved chunk file. "
            "Pass the user question and a single file path under /retrieved/."
        ),
        "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
        "model": ChatOpenRouter(model=ANALYST_MODEL, api_key=settings.openrouter_api_key),
    }

    return create_deep_agent(
        model=ChatOpenRouter(model=MAIN_MODEL, api_key=settings.openrouter_api_key),
        tools=[search_documents],
        backend=backend,
        system_prompt=INSTRUCTIONS,
        subagents=[chunk_analyst],
        # remembers each conversation (messages and files) by its thread_id
        checkpointer=InMemorySaver(),
    )
