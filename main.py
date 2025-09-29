import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, Optional, Literal
from typing_extensions import TypedDict


from langchain_mistralai.chat_models import ChatMistralAI
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# Setting environment variables

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is not set in the environment variables.")
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
llm_api = ChatMistralAI(model="mistral-small-2506", temperature=0) # for testing now I am using small model

print("Initialized Mistral LLM API.")

# --- Graph State Definition ---

class UserProfile(TypedDict, total=False):
    """The complete data structure for the visitor's preferences."""
    # Section core context
    language: str
    party_adults_count: int
    party_children_count: int
    mobility_needs: Literal["none", "stroller", "wheelchair", "limited_walking"]
    visit_date: str
    duration_hours: float
    visitor_familiarity: Literal["first_time", "know_a_bit", "frequent"]

    # Section Interests (simplified)
    interests_summary: str # A summary of interests instead of individual scores to start
    
    # Section constraints
    crowd_tolerance: Literal["low", "medium", "high"]
    weather_tolerance: Literal["rain_ok", "sun_ok", "prefer_indoor"]
    must_see: List[str]
    
    # Section completeness helper
    missing_fields: List[str]

# The action plan state 
class PlanStep(TypedDict):
    agent: str
    params: Dict[str, Any]


class GraphState(TypedDict):
    """The global state of the application, now including itinerary stages."""
    messages: Annotated[list, add_messages]
    user_profile: UserProfile
    
    # --- Itinerary Planning Stages ---
    raw_itinerary: Optional[Dict]
    verified_itinerary: Optional[Dict]
    enriched_itinerary: Optional[Dict]

    # --- Final Output ---
    narrative_response: str

def information_extractor_node(state: GraphState) -> dict:
    """
    Extracts information from the conversation to fill the user profile.
    This is the first "brain" that structures the request.
    """
    print("--- NODE: Information Extractor ---")
    
    # For this node, we use the local LLM that guarantees JSON output.
    llm = llm_api
    
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state['messages']])
    
    # The user profile as it currently exists. It will be empty on the first pass.
    current_profile = state.get('user_profile', {})
    current_profile_str = json.dumps(current_profile)

    # The complete schema we want to fill.
    schema_description = """
    {
        "duration_hours": "float - The duration of the visit in hours.",
        "mobility_needs": "string - One of 'none', 'stroller', 'wheelchair', 'limited_walking'.",
        "interests_summary": "string - A one-sentence summary of the user's interests.",
        "party_adults_count": "int - Number of adults in the party.",
        "party_children_count": "int - Number of children in the party.",
        "visitor_familiarity": "string - One of 'first_time', 'know_a_bit', 'frequent'.",
        "missing_fields": "list[str] - A list of the most important keys that are still missing."
    }
    """

    prompt = f"""
    You are an expert assistant for understanding visitor needs at the Palace of Versailles.
    Your task is to fill out a user profile in JSON format.

    Here is the conversation history:
    ---
    {conversation_history}
    ---
    
    Here is the user profile that has already been filled:
    ---
    {current_profile_str}
    ---
    
    Here is the complete JSON schema you must fill:
    ---
    {schema_description}
    ---
    
    Instructions:
    1. Read the conversation and update the profile with ALL the information you can infer.
    2. Keep existing information unless it's being explicitly corrected.
    3. If essential information (duration_hours, mobility_needs, interests_summary, party_adults_count, party_children_count) is still missing, add its key name to the "missing_fields" list.
    4. Return ONLY the complete, updated JSON object. Do not add any comments, explanations, or formatting.
    """
    
    print("   > Sending extraction prompt to LLM...")
    try:
        response = llm.invoke(prompt)
        updated_profile = json.loads(response.content)
        print(f"   > User profile updated: {json.dumps(updated_profile, indent=2)}")
        return {"user_profile": updated_profile}
    except json.JSONDecodeError as e:
        print(f"   > Error: Extractor did not return valid JSON: {e}")
        print(f"   > Response was: {response.content}")
        # Return current profile unchanged if parsing fails
        return {"user_profile": current_profile}
    except Exception as e:
        print(f"   > Unexpected error in extractor: {e}")
        return {"user_profile": current_profile}

def ask_clarification_question_node(state: GraphState) -> dict:
    """
    If information is missing, this node generates a question for the user.
    """
    print("--- NODE: Clarification Question ---")
    llm = llm_api
    
    missing_fields = state['user_profile'].get('missing_fields', [])
    if not missing_fields:
         print("   > No missing information, nothing to do.")
         return {}

    # Map field names to user-friendly descriptions
    field_descriptions = {
        "duration_hours": "how long you'd like to spend at Versailles",
        "mobility_needs": "any mobility requirements (wheelchair, stroller, limited walking)",
        "interests_summary": "what aspects of Versailles interest you most",
        "party_adults_count": "how many adults will be visiting",
        "party_children_count": "how many children will be visiting",
        "visitor_familiarity": "whether this is your first visit or you've been before"
    }
    
    missing_descriptions = [field_descriptions.get(field, field) for field in missing_fields]

    prompt = f"""
    You are the Royal Guide of Versailles, speaking warmly and professionally to a visitor.
    You need to gather some additional information to plan the perfect visit.
    
    Missing information: {', '.join(missing_descriptions)}
    
    Generate a friendly, engaging question that asks for this information naturally.
    Be conversational and helpful. Start with something like "To help me create the perfect visit for you..." or "I'd love to know a bit more about..."
    
    Return only the question text, no JSON formatting or extra content.
    """
    
    print(f"   > Asking LLM to generate a question for: {missing_fields}")
        
    try:
        response = llm.invoke(prompt)
        question = response.content.strip()
        print(f"   > Question generated: {question}")
        
        # We add the AI's question to the message history
        return {"messages": [AIMessage(content=question)]}
    except Exception as e:
        print(f"   > Error generating question: {e}")
        fallback_question = "Could you tell me a bit more about your planned visit to help me assist you better?"
        return {"messages": [AIMessage(content=fallback_question)]}

# --- Phase 1 Conditional Edge ---

def decide_next_step(state: GraphState) -> Literal["ask_clarification_question", "END"]:
    """
    Decides whether to ask for more info or to proceed with planning the tour.
    """
    print("--- ROUTER: Deciding Next Step ---")
    
    user_profile = state.get('user_profile', {})
    missing_fields = user_profile.get('missing_fields', [])
    
    if not user_profile or missing_fields:
        # If info is missing or the profile doesn't exist, ask for clarification.
        print(f"   > Decision: Profile incomplete. Missing: {missing_fields}. Asking for clarification.")
        return "ask_clarification_question"
    else:
        # Otherwise, the profile is complete, and we can move to the next phase.
        print("   > Decision: Profile complete. Proceeding to end (later will be planning).")
        return "END" 
        # return "itinerary_generation_node" # This will be the future destination

# Create checkpointer and build graph
checkpointer = InMemorySaver()
builder = StateGraph(GraphState)

# Add Phase 1 nodes
builder.add_node("information_extractor", information_extractor_node)
builder.add_node("ask_clarification_question", ask_clarification_question_node)

# The entry point is always information extraction
builder.set_entry_point("information_extractor")

# Add our main conditional edge
builder.add_conditional_edges(
    "information_extractor",
    decide_next_step,
    {
        "ask_clarification_question": "ask_clarification_question",
        "END": END 
    }
)

# After asking a question, we loop back to re-evaluate the situation
builder.add_edge("ask_clarification_question", "information_extractor")

# Compile the graph WITH the checkpointer.
le_guide_royal = builder.compile(checkpointer=checkpointer)
print("\n--- Phase 1 Graph Compiled Successfully with Memory ---")


# =============================================================================
# APPLICATION EXECUTION (INTERACTIVE) - ENHANCED
# =============================================================================

def display_welcome_message():
    """Display a welcoming message from the Royal Guide."""
    welcome_msg = """
🏰 ═══════════════════════════════════════════════════════════════ 🏰
    Welcome to the Palace of Versailles Virtual Guide!
    I am your Royal Guide, here to help you plan the perfect visit.
    
    Please tell me about your upcoming visit - I'd love to know:
    • How long you're planning to stay
    • Who will be joining you (adults, children)
    • What interests you most about Versailles
    • Any special requirements or preferences
    
    Type 'quit' or 'exit' to end our conversation.
🏰 ═══════════════════════════════════════════════════════════════ 🏰
    """
    print(welcome_msg)

def display_final_summary(user_profile):
    """Display a nice summary of the collected information."""
    print("\n" + "="*60)
    print("🎭 VISIT SUMMARY - Information Collected 🎭")
    print("="*60)
    
    for key, value in user_profile.items():
        if key != 'missing_fields' and value:
            formatted_key = key.replace('_', ' ').title()
            print(f"• {formatted_key}: {value}")
    
    print("="*60)
    print("✨ Perfect! I now have everything needed to plan your royal visit! ✨")
    print("(In the next phase, I would generate a personalized itinerary)")
    print("="*60)

if __name__ == "__main__":
    import uuid

    print("\n\n--- STARTING INTERACTIVE CHAT WITH THE ROYAL GUIDE ---")
    
    # Display welcome message
    display_welcome_message()
    
    # A unique ID for this specific conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    conversation_active = True

    while conversation_active:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n🏰 Royal Guide: Au revoir! May your future visit to Versailles be magnifique!")
                break
            
            if not user_input:  # Skip empty inputs
                continue

            print("\n🤔 Processing your message...")
            
            # Stream the graph execution
            events = le_guide_royal.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config, 
                stream_mode="values"
            )
            
            # Process events and display AI responses
            ai_responded = False
            for event in events:
                # Check if there are new messages in this event
                if "messages" in event and event["messages"]:
                    # Get the last message if it's from AI
                    last_message = event["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        print(f"\n🏰 Royal Guide: {last_message.content}")
                        ai_responded = True

            # Check if the conversation has reached completion
            final_snapshot = le_guide_royal.get_state(config)
            if not final_snapshot.next:
                print("\n🎉 Information gathering complete!")
                display_final_summary(final_snapshot.values.get('user_profile', {}))
                conversation_active = False
            elif ai_responded:
                # AI has asked a question, continue the loop to wait for user input
                continue

        except KeyboardInterrupt:
            print("\n\n🏰 Royal Guide: Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Let's try continuing our conversation...")
            continue