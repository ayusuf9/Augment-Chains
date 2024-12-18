from typing import Dict, List, Optional, Any
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import re

# Define our base state type
class AgentState:
    messages: List[str]
    current_status: str
    analysis_result: Optional[Dict[str, Any]]
    
    def __init__(self):
        self.messages = []
        self.current_status = "STARTING"
        self.analysis_result = None

# Create our analysis components
def create_analysis_prompt(text: str) -> str:
    return f"""You are a Federal Reserve policy expert. Analyze the following speech text and 
    determine if it is dovish (indicating easier monetary policy) or hawkish (indicating tighter monetary policy).
    
    Consider these aspects:
    1. Language about inflation concerns
    2. References to economic growth
    3. Discussion of employment conditions
    4. Forward guidance language
    5. Risk assessment terminology
    
    Speech text:
    {text}
    
    Classify this as either DOVISH or HAWKISH and provide your reasoning.
    Format your response as:
    CLASSIFICATION: [DOVISH/HAWKISH]
    CONFIDENCE: [HIGH/MEDIUM/LOW]
    REASONING: [Your detailed analysis]
    """

def analyze_speech_content(state: AgentState) -> AgentState:
    """Core analysis node that processes speech content"""
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    
    # Get the latest message which should contain our speech text
    speech_text = state.messages[-1]
    
    # Create and send our analysis prompt
    response = llm.invoke([HumanMessage(content=create_analysis_prompt(speech_text))])
    
    # Parse the response
    response_text = response.content
    
    # Extract classification and confidence
    classification_match = re.search(r'CLASSIFICATION:\s*(DOVISH|HAWKISH)', response_text)
    confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response_text)
    reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.DOTALL)
    
    state.analysis_result = {
        "classification": classification_match.group(1) if classification_match else "UNKNOWN",
        "confidence": confidence_match.group(1) if confidence_match else "LOW",
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
    }
    
    state.current_status = "COMPLETED"
    return state

def create_fed_speech_analyzer() -> Graph:
    """Creates and returns our speech analysis workflow"""
    
    # Create our workflow graph
    workflow = StateGraph(AgentState)
    
    # Add our analysis node
    workflow.add_node("analyze", analyze_speech_content)
    
    # Define the workflow edges
    workflow.set_entry_point("analyze")
    
    # Conditional edge: if status is COMPLETED, we're done
    workflow.add_conditional_edges(
        "analyze",
        lambda x: x.current_status == "COMPLETED",
        {
            True: "end",
            False: "analyze"  # Shouldn't happen but good practice
        }
    )
    
    return workflow.compile()

# Helper function to run the analysis
def analyze_fed_speech(speech_text: str) -> Dict[str, Any]:
    """
    Analyzes a Federal Reserve speech and returns the classification results
    
    Args:
        speech_text (str): The full text of the speech to analyze
        
    Returns:
        Dict containing classification, confidence, and reasoning
    """
    # Initialize our state
    initial_state = AgentState()
    initial_state.messages.append(speech_text)
    
    # Create and run our workflow
    workflow = create_fed_speech_analyzer()
    final_state = workflow.invoke(initial_state)
    
    return final_state.analysis_result