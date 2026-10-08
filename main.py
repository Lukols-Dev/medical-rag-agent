import argparse
import uuid

from langchain.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from agent import build_agent

EXIT_COMMANDS = {"exit", "quit", "q"}
NEW_CHAT_COMMAND = "new"


def new_thread() -> dict:
    """
    Config for a fresh conversation - the agent remembers messages per thread_id.
    """
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def describe_tool_call(call: dict) -> str:
    """
    Short, human-readable description of what the agent is doing.
    """
    args = call["args"]
    if call["name"] == "search_documents":
        return f"searching: {args.get('query', '')}"
    if call["name"] == "task":
        return f"delegating to {args.get('subagent_type', 'subagent')}"
    return call["name"]


def ask(agent: CompiledStateGraph, question: str, config: dict) -> str:
    """
    Send a question to the agent, print its steps while it works and return the final answer.
    """
    updates = agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates",
    )
    for update in updates:
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue
            for message in node_update.get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        print(f"  -> {describe_tool_call(call)}")

    return agent.get_state(config).values["messages"][-1].text


def chat(agent: CompiledStateGraph) -> None:
    """
    Interactive chat in the terminal.
    """
    config = new_thread()
    print(
        "Ask about Bell's palsy. "
        f"Type '{NEW_CHAT_COMMAND}' to start a new conversation, 'exit' or Ctrl+D to quit."
    )

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break
        if question.lower() == NEW_CHAT_COMMAND:
            config = new_thread()
            print("Started a new conversation.")
            continue

        try:
            answer = ask(agent, question, config)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            continue
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            # keep the chat running after API errors, e.g. rate limits on free models
            print(f"\nError: {error}")
            continue

        print(f"\nAgent: {answer}")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about the indexed medical articles.")
    parser.add_argument(
        "question",
        nargs="*",
        help="ask a single question and exit; without it an interactive chat starts",
    )
    args = parser.parse_args()

    agent = build_agent()
    if args.question:
        print(ask(agent, " ".join(args.question), new_thread()))
    else:
        chat(agent)


if __name__ == "__main__":
    main()
