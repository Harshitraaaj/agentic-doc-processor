"""
LangGraph Visualization Utilities
"""
from utils.logger import logger


class GraphVisualizer:
    """
    Utility class for LangGraph visualization
    
    Provides:
    1. Graph structure visualization (Mermaid diagrams)
    2. Execution trace visualization
    """
    
    def __init__(self):
        """Initialize visualizer"""
        logger.info("GraphVisualizer initialized")

    def _normalize_agent_to_node(self, agent_name: str) -> str:
        """Map runtime agent names to workflow node names."""
        mapping = {
            "ClassifierAgent": "classify",
            "ExtractorAgent": "extract",
            "ValidatorAgent": "validate",
            "SelfRepairNode": "repair",
            "RedactorAgent": "redact",
            "ReporterAgent": "report",
            "HumanInLoopAgent": "human_review",
            "SupervisorAgent": "supervisor",
        }
        if agent_name in mapping:
            return mapping[agent_name]

        normalized = (agent_name or "unknown").replace("Agent", "").replace("Node", "").lower()
        if "repair" in normalized:
            return "repair"
        return normalized

    def _infer_supervisor_path(self, base_path: list[str]) -> list[str]:
        """
        Insert supervisor/HITL checkpoint nodes into a runtime path for better alignment
        with the current supervisor orchestration graph.
        """
        if not base_path:
            return base_path

        inferred = []
        for index, node in enumerate(base_path):
            inferred.append(node)
            next_node = base_path[index + 1] if index + 1 < len(base_path) else None

            if node == "classify":
                inferred.append("supervise_classification")
                if next_node == "extract":
                    inferred.append("human_review_classify")

            if node == "validate":
                inferred.append("supervise_validation")

        # De-duplicate consecutive repeats caused by repair loops
        cleaned = []
        for node in inferred:
            if not cleaned or cleaned[-1] != node:
                cleaned.append(node)
        return cleaned
    
    def generate_mermaid_diagram(self, workflow) -> str:
        """
        Generate Mermaid diagram from LangGraph workflow
        
        Args:
            workflow: LangGraph workflow/graph instance (DocumentProcessingWorkflow)
        
        Returns:
            Mermaid diagram as string
        """
        try:
            # Check if it's our custom workflow class
            if hasattr(workflow, 'get_graph_mermaid'):
                mermaid = workflow.get_graph_mermaid()
                return mermaid
            
            # Try compiled graph (standard LangGraph)
            elif hasattr(workflow, 'compiled_graph') and workflow.compiled_graph:
                graph = workflow.compiled_graph.get_graph()
                mermaid = graph.draw_mermaid()
                return mermaid
            
            # Try graph attribute
            elif hasattr(workflow, 'graph'):
                # Compile if needed
                if not hasattr(workflow, 'compiled_graph') or workflow.compiled_graph is None:
                    if hasattr(workflow, 'compile'):
                        workflow.compile()
                
                if workflow.compiled_graph:
                    graph = workflow.compiled_graph.get_graph()
                    mermaid = graph.draw_mermaid()
                    return mermaid
            
            # Direct StateGraph
            elif hasattr(workflow, 'get_graph'):
                graph = workflow.get_graph()
                mermaid = graph.draw_mermaid()
                return mermaid
            
            # Fallback
            raise AttributeError("Cannot find graph visualization method")
            
        except Exception as e:
            logger.error(f"Failed to generate Mermaid diagram: {e}")
            return f"graph TD\n  Error[Error: {str(e)}]"
    
    def extract_execution_path(self, trace_log: list, mode: str = "auto") -> list:
        """
        Extract the actual execution path from trace log
        
        Args:
            trace_log: List of ResponsibleAILog entries from state
        
        Returns:
            List of node names in execution order
        """
        path = []
        for entry in trace_log:
            agent = entry.get("agent_name", "unknown")
            if agent and agent not in ["START", "END"]:
                path.append(self._normalize_agent_to_node(agent))

        if not path:
            return path

        mode_normalized = (mode or "auto").strip().lower()
        if mode_normalized == "supervisor":
            return self._infer_supervisor_path(path)
        if mode_normalized == "standard":
            return path

        # auto mode: infer supervisor checkpoints when path looks like full orchestration flow
        if "classify" in path and "validate" in path and "report" in path:
            return self._infer_supervisor_path(path)
        return path
    
    def generate_execution_path_diagram(self, trace_log: list, mode: str = "auto") -> str:
        """
        Generate Mermaid diagram showing ONLY the actual execution path taken
        
        Args:
            trace_log: List of ResponsibleAILog entries from state
        
        Returns:
            Mermaid diagram with only executed nodes (no gray boxes)
        """
        path = self.extract_execution_path(trace_log, mode=mode)
        
        # Start diagram
        diagram = ["graph TD"]
        diagram.append("    classDef executed fill:#90EE90,stroke:#006400,stroke-width:3px,color:#000")
        diagram.append("")
        
        # Only show executed nodes (no gray boxes)
        executed_nodes = list(dict.fromkeys(path))  # Remove duplicates, preserve order
        
        # Add START
        diagram.append("    START([🚀 START]):::executed")
        
        # Add only executed nodes with emojis
        node_emojis = {
            "classify": "📋",
            "extract": "📤",
            "validate": "✓",
            "repair": "🔧",
            "redact": "🔒",
            "report": "📊",
            "supervise_classification": "🧠",
            "supervise_validation": "🧠",
            "human_review_classify": "👤",
            "human_review_extract": "👤",
            "human_review": "👤",
        }
        
        for node in executed_nodes:
            emoji = node_emojis.get(node, "")
            node_label = f"{emoji} {node.capitalize()}"
            diagram.append(f"    {node}[{node_label}]:::executed")
        
        # Add END
        diagram.append("    END([✔️ END]):::executed")
        diagram.append("")
        
        # Add edges based on execution path
        if path:
            # Connect START to first node
            diagram.append(f"    START --> {path[0]}")
            
            # Add edges for actual path (skip first since we just connected START)
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]
                
                # Style arrows based on path
                if current == "validate" and next_node == "repair":
                    diagram.append(f"    {current} -.->|needs repair| {next_node}")
                elif current == "repair" and next_node == "validate":
                    diagram.append(f"    {current} -.->|retry| {next_node}")
                elif current == "supervise_validation" and next_node == "repair":
                    diagram.append(f"    {current} -.->|retry decision| {next_node}")
                elif current == "supervise_validation" and next_node == "redact":
                    diagram.append(f"    {current} -->|auto approved| {next_node}")
                elif current == "supervise_classification" and next_node == "human_review_classify":
                    diagram.append(f"    {current} -->|mandatory HITL| {next_node}")
                else:
                    diagram.append(f"    {current} --> {next_node}")
            
            # Connect last node to END
            diagram.append(f"    {path[-1]} --> END")
        
        return "\n".join(diagram)
    
    def visualize_execution_trace(self, trace_log: list, mode: str = "auto") -> str:
        """
        Generate Mermaid sequence diagram from actual execution trace
        
        Args:
            trace_log: List of ResponsibleAILog entries from state
        
        Returns:
            Mermaid sequence diagram showing actual execution flow
        """
        path = self.extract_execution_path(trace_log, mode=mode)

        diagram = ["sequenceDiagram"]
        diagram.append("    participant User")
        diagram.append("    participant Workflow")

        participants = []
        for node in path:
            if node not in participants:
                participants.append(node)
                safe_name = node.replace("-", "_")
                diagram.append(f"    participant {safe_name}")

        diagram.append("")
        diagram.append("    User->>Workflow: Submit Document")

        prev = "Workflow"
        for node in path:
            safe_name = node.replace("-", "_")
            action = node.replace("_", " ").title()
            diagram.append(f"    {prev}->>+{safe_name}: {action}")
            diagram.append(f"    {safe_name}-->>-{prev}: Done")
            prev = safe_name

        diagram.append(f"    {prev}->>User: Return Result")
        return "\n".join(diagram)


# Global visualizer instance
graph_visualizer = GraphVisualizer()
