from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate


class ObjectOfInterest(BaseModel):
    object_id: str
    title: str
    facts: List[str]

class Place(BaseModel):
    place_id: str
    title: str
    priority: int = Field(..., description="The visit order for this place WITHIN its subgroup. Lower numbers come first.")
    objects: List[ObjectOfInterest]

class Subgroup(BaseModel):
    subgroup_id: str
    title: str
    priority: int = Field(..., description="The visit order for this subgroup WITHIN its parent group. Lower numbers come first.")
    places: List[Place]

class Group(BaseModel):
    group_id: str
    title: str
    priority: int = Field(..., description="The visit order for this main group (e.g., Château vs. Jardins). Lower numbers come first.")
    duration_minutes: int
    subgroups: List[Subgroup]

class ItineraryOption(BaseModel):
    option_id: str
    title: str
    groups: List[Group]

class ItineraryOutput(BaseModel):
    language: str
    user_objectives: List[str]
    options: List[ItineraryOption]



planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class tour planner for the Palace of Versailles. Your goal is to create a personalized and logically sequenced itinerary based on a user's profile and a list of suggested locations.

You must generate the itinerary in a structured JSON format that adheres to the provided schema.

Your most important task is to assign a hierarchical `priority` at three levels:
1.  **Group Priority**: Determines the overall order of the main sections of the visit (e.g., 1. Le Château, 2. Les Jardins).
2.  **Subgroup Priority**: Determines the order of visit within a Group (e.g., within Le Château: 1. Grand Appartement du Roi, 2. Galerie des Glaces).
3.  **Place Priority**: Determines the order of visit within a Subgroup.

For all priority fields, a lower number means it should be visited earlier (e.g., priority 1 comes before priority 2).

Analyze the user's objectives (e.g., 'avoid crowds', 'focus on mythology', 'relaxed pace') and the provided suggestions to create the most logical and enjoyable sequence. For example, if a user wants to avoid crowds, you might prioritize the Gardens in the morning and the Château in the afternoon."""
        ),
        (
            "human",
            "User Profile: {profile}\n\nSuggested Locations: {suggestions}\n\nPlease generate the full, structured itinerary with hierarchical priorities."
        ),
    ]
)


# final_state is the output from your LangGraph agent
itinerary_data = final_state['itinerary']

def planner_agent(state: AppState) -> dict:
    """
    This agent now generates a hierarchically prioritized itinerary.
    """
    print("--- 3. PLANNING ITINERARY WITH HIERARCHICAL PRIORITY ---")

    structured_llm = llm.with_structured_output(ItineraryOutput)
    planner_chain = planner_prompt | structured_llm

    itinerary_response = planner_chain.invoke(
        {"profile": state["profile"], "suggestions": state["suggestions"]}
    )

    return {"itinerary": itinerary_response.dict()}

for option in itinerary_data['options']:
    print(f"Option: {option['title']}")
    
    # Sort groups by their priority
    sorted_groups = sorted(option['groups'], key=lambda g: g['priority'])
    
    for group in sorted_groups:
        print(f"  - Group: {group['title']} (Duration: {group['duration_minutes']} mins)")
        
        # Sort subgroups by their priority
        sorted_subgroups = sorted(group['subgroups'], key=lambda sg: sg['priority'])
        
        for subgroup in sorted_subgroups:
            print(f"    - Subgroup: {subgroup['title']}")
            
            # Sort places by their priority
            sorted_places = sorted(subgroup['places'], key=lambda p: p['priority'])
            
            for place in sorted_places:
                print(f"      - Visit: {place['title']}")
                for obj in place['objects']:
                    print(f"        - See: {obj['title']}")