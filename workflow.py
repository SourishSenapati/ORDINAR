"""Module defining the LangGraph orchestration workflow for agentic scientific discovery."""
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from agents import DataArchitectAgent, TheoristAgent, CriticAgent


class ResearchState(TypedDict):
    """The synchronized memory of the Agent Swarm."""
    csv_input: str
    processed_data: Dict[str, Any]
    candidate_model: Dict[str, Any]
    audit_report: Dict[str, Any]
    alpha_parameters: List[float]
    iteration_count: int
    discovery_successful: bool

# Node implementations


def data_ingestion_layer(state: ResearchState):
    """Orchestrates data ingestion and noise profiling."""
    agent = DataArchitectAgent()
    data = agent.prepare_state(state['csv_input'])
    return {"processed_data": data, "iteration_count": state['iteration_count'] + 1}


def mechanism_discovery_layer(state: ResearchState):
    """Executes discrete discovery of interaction matrices."""
    agent = TheoristAgent()
    # Pull the latest alpha estimate from history
    alpha = state['alpha_parameters'][-1]
    model = agent.execute_discovery(state['processed_data'], alpha)
    return {"candidate_model": model}


def validation_critical_layer(state: ResearchState):
    """Performs stability and RMSE audit to validate the discovered mechanics."""
    agent = CriticAgent()
    report = agent.audit_model(
        state['candidate_model'], state['processed_data'])

    # Success condition (e.g., score > 0.8)
    success = report['score'] > 0.8
    return {"audit_report": report, "discovery_successful": success}


def router_logic(state: ResearchState):
    """Controls the cyclic reasoning loop based on Critic's report."""
    if state['discovery_successful'] or state['iteration_count'] > 15:
        return "complete"
    return "refine"


def refinement_logic(state: ResearchState):
    """Adjusts hyperparameters (like Alpha) to escape local minima if fit is poor."""
    old_alpha = state['alpha_parameters'][-1]
    # Simple strategy: decrease alpha slightly if fit isn't good
    new_alpha = old_alpha * 0.98 if old_alpha > 0.05 else 0.15
    return {
        "alpha_parameters": state['alpha_parameters'] + [new_alpha],
        "iteration_count": state['iteration_count'] + 1
    }


def build_research_workflow():
    """Compiles the Agentic State Machine into a runnable Research Processor."""
    builder = StateGraph(ResearchState)

    # Define Nodes
    builder.add_node("ingest", data_ingestion_layer)
    builder.add_node("discover", mechanism_discovery_layer)
    builder.add_node("validate", validation_critical_layer)
    builder.add_node("refine", refinement_logic)

    # Entry Point
    builder.set_entry_point("ingest")

    # Static Edges
    builder.add_edge("ingest", "discover")
    builder.add_edge("discover", "validate")

    # Conditional Feedback Loop
    builder.add_conditional_edges(
        "validate",
        router_logic,
        {
            "complete": END,
            "refine": "refine"
        }
    )
    builder.add_edge("refine", "discover")

    return builder.compile()
