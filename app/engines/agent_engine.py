import json
import re
from typing import List, Dict, Any, Tuple, Callable
from app.core.llm import LLMService
from app.db.database_manager import DatabaseManager
from app.db.retrieval_manager import HierarchicalRAGRetriever
from app.engines.semantic_chunker import SemanticEngine
from app.core.config import settings

class ReActAgent:
    """
    Implements a Reason-and-Act (ReAct) loop.
    Enables iterative search, selection of tools, and self-correction.
    """
    def __init__(self, db_manager: DatabaseManager, vector_engine: SemanticEngine):
        self.db_manager = db_manager
        self.vector_engine = vector_engine
        self.llm = LLMService()
        
        # Register available tools
        self.tools: Dict[str, Tuple[Callable, str]] = {
            "list_documents": (
                self._tool_list_documents, 
                "list_documents() - Returns a list of all documents currently ingested. No inputs required. Input format: {}"
            ),
            "search_all_documents": (
                self._tool_search_all_documents, 
                "search_all_documents(query: str) - Performs semantic vector search across all documents. Input format: {\"query\": \"your search query\"}"
            ),
            "search_document": (
                self._tool_search_document, 
                "search_document(query: str, file_id: str) - Performs semantic vector search on a specific document. file_id can be the document's unique hash ID or its title. Input format: {\"query\": \"your search query\", \"file_id\": \"document_id_or_title\"}"
            ),
            "read_chunk": (
                self._tool_read_chunk, 
                "read_chunk(chunk_id: str) - Fetches the full raw text content of a specific chunk by its ID. Input format: {\"chunk_id\": \"chunk_id_here\"}"
            )
        }

    # --- Tool Implementations ---
    async def _tool_list_documents(self, args: Dict[str, Any]) -> str:
        docs = await self.db_manager.list_all_documents()
        if not docs:
            return "No documents found in the database. Please tell the user to upload files."
        return json.dumps(docs, indent=2)

    async def _tool_search_all_documents(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: Missing 'query' parameter."
        async with self.db_manager.SessionLocal() as session:
            retriever = HierarchicalRAGRetriever(session, self.vector_engine)
            results = await retriever.retrieve_across_all_documents(query, top_k=4)
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk_id": r["chunk_id"],
                "document_title": r["document_title"],
                "text": r["text"],
                "similarity_score": f"{r['similarity']:.2%}"
            })
        return json.dumps(formatted_results, indent=2)

    async def _tool_search_document(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        file_id = args.get("file_id")
        if not query or not file_id:
            return "Error: Both 'query' and 'file_id' parameters are required."
            
        # Resolve the identifier (which could be the hash ID or the document title)
        resolved_file_id = await self.db_manager.resolve_file_id(file_id)
        if not resolved_file_id:
            return f"Error: Document matching identifier '{file_id}' not found. Please run 'list_documents' to see valid document titles and IDs."
            
        async with self.db_manager.SessionLocal() as session:
            retriever = HierarchicalRAGRetriever(session, self.vector_engine)
            # Use the resolved hash ID for similarity search
            results = await retriever.retrieve_relevant_chunks(query, resolved_file_id, top_k=4)
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "similarity_score": f"{r['similarity']:.2%}"
            })
        return json.dumps(formatted_results, indent=2)

    async def _tool_read_chunk(self, args: Dict[str, Any]) -> str:
        chunk_id = args.get("chunk_id")
        if not chunk_id:
            return "Error: Missing 'chunk_id' parameter."
        chunk = await self.db_manager.fetch_chunk_by_id(chunk_id)
        if not chunk:
            return f"Error: Chunk with ID '{chunk_id}' not found."
        return json.dumps(chunk, indent=2)

    # --- Agent Orchestration Loop ---
    def _build_system_instruction(self) -> str:
        # Prompt telling the LLM how to format outputs and use tools
        tool_descriptions = "\n".join([f"- {desc}" for name, (_, desc) in self.tools.items()])
        
        return f"""You are an Agentic Retrieval-Augmented Generation (RAG) assistant.
        You have access to a local knowledge base of documents. You can run searches and query documents using tools.

        To answer user questions, you MUST follow a structured reasoning loop:
        1. Thought: Reason about what information you need and decide which tool to call.
        2. Action: Choose a tool to run. MUST be one of: [{', '.join(self.tools.keys())}].
        3. Action Input: A JSON object containing the parameters for the chosen tool.
        4. (The system will execute the tool and provide you with an Observation)
        5. Repeat until you have enough information to write the Final Answer.

        Format your output EXACTLY like this:
        Thought: your reasoning here
        Action: tool_name
        Action Input: {{"param_name": "value"}}

        Once you have gathered enough information to fully answer the user, write:
        Thought: I have gathered all necessary information.
        Final Answer: your comprehensive response based on the search results. Include citations referencing the document title or chunk ID.

        Rules:
        - If a tool returns no results, try a different search query or list documents to see what is available.
        - Do not make up facts. Only answer based on observations.
        - Always output either an Action block or a Final Answer block. Never both at the same time.

        Available Tools:
        {tool_descriptions}
        """

    async def run(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs the ReAct loop.
        Returns:
            Tuple[str, List[dict]]: (Final Answer, List of reasoning steps/history)
        """
        history: List[Dict[str, Any]] = []
        conversation_context = f"User Question: {query}\n"

        for iteration in range(settings.agent_max_iterations):
            # 1. Ask LLM for the next step
            system_instruction = self._build_system_instruction()
            response = await self.llm.generate_response(system_instruction, conversation_context)
            
            # Save step to history for CLI / UI presentation
            step_record = {"iteration": iteration, "llm_raw_response": response}
            history.append(step_record)

            # 2. Parse Thought, Action, Action Input, or Final Answer
            action_match = re.search(r"Action:\s*(\w+)", response)
            input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
            final_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)

            if final_match:
                # The agent finished!
                step_record["type"] = "final_answer"
                step_record["content"] = final_match.group(1).strip()
                return final_match.group(1).strip(), history

            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input_str = input_match.group(1).strip()
                
                step_record["type"] = "action"
                step_record["tool"] = tool_name
                step_record["input"] = tool_input_str
                
                # Parse tool arguments
                try:
                    tool_args = json.loads(tool_input_str)
                except json.JSONDecodeError:
                    observation = f"Error: Action Input is not valid JSON: '{tool_input_str}'"
                    step_record["observation"] = observation
                    conversation_context += f"\n{response}\nObservation: {observation}\n"
                    continue
                
                # Execute tool
                if tool_name in self.tools:
                    tool_func = self.tools[tool_name][0]
                    try:
                        observation = await tool_func(tool_args)
                    except Exception as e:
                        observation = f"Error running tool '{tool_name}': {str(e)}"
                else:
                    observation = f"Error: Tool '{tool_name}' is not registered."
                
                step_record["observation"] = observation
                
                # Append action and observation back to the context
                conversation_context += f"\n{response}\nObservation: {observation}\n"
                
            else:
                # Malformed output from LLM; guide it to try again
                observation = "Error: Your output did not follow the required format. You must output 'Action:' and 'Action Input:' OR 'Final Answer:'."
                step_record["type"] = "malformed"
                step_record["observation"] = observation
                conversation_context += f"\n{response}\nObservation: {observation}\n"

        # Max iterations reached
        timeout_msg = "Agent timed out. I was unable to retrieve sufficient information in time to answer your question."
        history.append({"iteration": settings.agent_max_iterations, "type": "timeout", "content": timeout_msg})
        return timeout_msg, history
