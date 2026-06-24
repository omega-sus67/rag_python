import argparse
import asyncio
import os
import sys
import json
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from app.controllers.main_controller import MainController
from fastapi import HTTPException

# Initialize global rich terminal console.
console = Console()

async def ingest_document(filepath: str):
    """
    Ingestion handler for CLI:
    1. Checks if the PDF filepath exists locally.
    2. Runs systems init and document parsing.
    3. Handles and logs error responses.
    """
    if not os.path.exists(filepath):
        console.print(f"[bold red]Error:[/] File not found at '{filepath}'")
        sys.exit(1)
        
    console.print(Panel(f"[bold cyan]RAG Pipeline[/] - Ingesting Document\n[dim]{filepath}[/]", border_style="cyan"))
    
    controller = MainController()
    
    # Use Rich Progress context to keep the terminal interactive during processing.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[yellow]Initializing system...", total=None)
        await controller.initialize_system()
        
        progress.update(task, description="[cyan]Parsing, chunking, and embedding document...")
        try:
            result = await controller.process_and_ingest_pdf(filepath)
            
            console.print(Panel(
                f"[bold green]Success![/]\n\n"
                f"[bold]Title:[/] {result['title']}\n"
                f"[bold]Document ID:[/] [yellow]{result['document_id']}[/]\n"
                f"[bold]Chunks Processed:[/] {result['chunks_processed']}",
                title="Ingestion Complete",
                border_style="green"
            ))
        except HTTPException as e:
            if e.status_code == 400 and "already exists" in e.detail:
                console.print(Panel("[bold yellow]Notice:[/] Document is already ingested in the database.", border_style="yellow"))
            else:
                console.print(f"[bold red]Pipeline Error:[/] {e.detail}")
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/] {str(e)}")

async def query_document(file_id: str, query: str, top_k: int):
    """
    Retrieval and presentation handler for CLI:
    1. Fetches document metadata.
    2. Runs vector similarity search.
    3. Prints colorized result panels based on similarity score ranges.
    """
    console.print(Panel(f"[bold cyan]RAG Pipeline[/] - Semantic Search\n[dim]Querying Document: {file_id}[/]", border_style="cyan"))
    
    controller = MainController()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[yellow]Initializing system...", total=None)
        await controller.initialize_system()
        
        progress.update(task, description=f"[cyan]Searching for top {top_k} matches...")
        
        try:
            # Check document existence before querying.
            doc = await controller.fetch_document(file_id)
            console.print(f"Searching in: [bold]{doc.title}[/]\n")
            
            results = await controller.query_document(query, file_id, top_k)
            
            if not results:
                console.print("[yellow]No relevant chunks found.[/]")
                return
                
            for i, res in enumerate(results, 1):
                sim_score = res['similarity']
                # Determine score coloring based on thresholds (green for strong match, red for weak).
                score_color = "green" if sim_score > 0.5 else "yellow" if sim_score > 0.3 else "red"
                
                content = Text()
                content.append(f"Similarity: ", style="bold")
                content.append(f"{sim_score:.2%}\n\n", style=f"bold {score_color}")
                content.append(res['text'])
                
                console.print(Panel(
                    content,
                    title=f"Result #{i} (Chunk ID: {res['chunk_id'][:8]}...)",
                    border_style="blue"
                ))
                
        except HTTPException as e:
            console.print(f"[bold red]API Error:[/] {e.detail}")
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/] {str(e)}")

async def start_agent_chat():
    """
    Launches an interactive shell enabling chat conversations with the ReAct Agent.
    """
    console.print(Panel(
        "[bold cyan]Agentic RAG Engine - Interactive Chat[/]\n"
        "Ask questions across all documents. The agent will show its reasoning steps.\n"
        "[dim]Type 'exit' or 'quit' to close.[/]", 
        border_style="cyan"
    ))
    
    controller = MainController()
    
    # 1. Prepare system
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("[yellow]Initializing system...", total=None)
        await controller.initialize_system()
        
    # 2. Main interactive input loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/]")
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting chat. Goodbye![/]")
                break
                
            if not user_input.strip():
                continue
                
            console.print("[dim]Thinking...[/]")
            
            # Execute Agent loop
            result = await controller.ask_agent(user_input)
            
            # Print reasoning steps step-by-step
            for step in result["steps"]:
                iter_num = step["iteration"] + 1
                
                # Parse thought text for display
                raw = step["llm_raw_response"]
                thought_match = re.search(r"Thought:\s*(.*)", raw)
                thought = thought_match.group(1).split("Action:")[0].strip() if thought_match else "Reasoning..."
                
                console.print(f"\n[bold yellow]Step {iter_num} - Thought:[/]")
                console.print(Panel(thought, border_style="yellow"))
                
                if step.get("type") == "action":
                    console.print(f"[bold magenta]Action: Calling Tool '{step['tool']}'[/]")
                    console.print(f"[magenta]Inputs:[/] [dim]{step['input']}[/]")
                    
                    # Highlight tool observation result
                    obs_snippet = step["observation"]
                    # Limit long observations in display
                    if len(obs_snippet) > 400:
                        obs_snippet = obs_snippet[:400] + "\n... (truncated for readability)"
                        
                    console.print(Panel(obs_snippet, title="Observation", border_style="purple"))
                    
            # Print Final Answer
            console.print("\n");
            console.print(Panel(
                result["answer"], 
                title="[bold green]Final Answer[/]", 
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting chat. Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
    
def main():
    """Entrypoint parsing CLI arguments and routing calls to respective async handlers."""
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command subcommand.
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document into the database")
    ingest_parser.add_argument("filepath", type=str, help="Path to the PDF file")
    agent_parser = subparsers.add_parser("agent", help="Start an interactive reasoning agent chat session")
    
    # Query command subcommand.
    query_parser = subparsers.add_parser("query", help="Query an ingested document")
    query_parser.add_argument("file_id", type=str, help="The Document ID to query")
    query_parser.add_argument("query_text", type=str, help="Your semantic search query")
    query_parser.add_argument("--top", type=int, default=3, help="Number of chunks to return (default: 3)")
    
    args = parser.parse_args()
    
    # Command routing using async run loops.
    if args.command == "ingest":
        asyncio.run(ingest_document(args.filepath))
    elif args.command == "query":
        asyncio.run(query_document(args.file_id, args.query_text, args.top))
    elif args.command == "agent":
        asyncio.run(start_agent_chat())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
