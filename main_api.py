# ÉTAPE 0 & 1 : IMPORTS, CONFIGURATION ET DÉFINITION DE L'ÉTAT
# =============================================================================
# --- Imports de la Bibliothèque Standard Python ---
import asyncio
import json
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# --- Imports pour le Typage (Typing) ---
from typing import (Any, Annotated, AsyncIterator, Dict, List, Literal,
                    Optional, Union)
from typing_extensions import TypedDict

# --- Imports de Bibliothèques Tierces ---
import requests
from dotenv import load_dotenv

# --- Imports de LangChain et LangGraph ---

# Modèles de Chat (LLMs)
from langchain_community.chat_models import ChatOllama
from langchain_mistralai.chat_models import ChatMistralAI

# Composants Core de LangChain
import langchain_core.documents
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# Intégrations (Embeddings, Vectorstores)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LangGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

# --- Imports Spécifiques à des Fournisseurs (API) ---
from mistralai import Mistral

# --- Imports Locaux de l'Application ---
from RAG_agent.RAGagent import RAGagent

rag_agent = RAGagent()

# Dictionnaire global pour stocker les historiques de chat par session
chats_by_session_id = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Récupère ou crée un historique de chat pour une session donnée."""
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

# --- Configuration ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is not set in the environment variables.")

# IDs des agents pré-configurés sur la plateforme Mistral
AGENT_ID_EXTRACTOR = os.getenv("MISTRAL_AGENT_ID_EXTRACTOR") 
AGENT_ID_QUESTIONER = os.getenv("MISTRAL_AGENT_ID_QUESTIONER") 

if not AGENT_ID_EXTRACTOR or not AGENT_ID_QUESTIONER:
    raise ValueError("Please set MISTRAL_AGENT_ID_EXTRACTOR and MISTRAL_AGENT_ID_QUESTIONER in your .env file.")

# Client Mistral
client = Mistral(api_key=MISTRAL_API_KEY)
print("Mistral client initialized.")
# Modèle LLM pour la planification directe
llm_planner_model = ChatMistralAI(model="mistral-large-latest", temperature=0.1)


# --- Définition de l'État ---
# Correspond à la sortie JSON de l'AGENT_ID_1
class UserProfile(TypedDict, total=False):
    language: str
    time_window: Dict
    interests: List[str]
    constraints: List[str]
    meals_services: List[str]
    tickets_logistics: List[str]
    missing_fields: List[str]

def merge_collected_data(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Dict[str, Any]:
    """Fusionne deux dictionnaires. Prend en charge les cas où l'un ou l'autre est None."""
    if left is None:
        return right or {}
    if right is None:
        return left or {}
    
    # Crée une copie pour ne pas modifier l'original
    merged = left.copy()
    merged.update(right)
    return merged

class GraphState(TypedDict):
    """L'état global de notre application de chat."""
    messages: Annotated[list, add_messages]
    user_profile: UserProfile
    question_attempts: int
    collected_data: Dict[str, Any]
    collected_data: Annotated[Dict[str, Any], merge_collected_data]




# =============================================================================
# ÉTAPE 2 & 3 : DÉFINITION DES NŒUDS (WRAPPERS D'AGENTS MISTRAL)
# =============================================================================

# Le nœud accepte maintenant 'config'
def information_extractor_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Ce nœud appelle l'Agent Mistral AGENT_ID_1 pour extraire les informations.
    Il utilise maintenant l'historique de conversation complet.
    """
    print("--- NODE: Information Extractor (calling Mistral Agent API) ---")
    
    # 1. Récupérer le session_id depuis la configuration
    session_id = config["configurable"]["session_id"]
    
    # 2. Récupérer l'historique de la conversation
    chat_history = get_chat_history(session_id)
    
    # 3. Le dernier message de l'utilisateur est dans l'état actuel
    last_user_message = state['messages'][-1]
    
    # 4. Préparer l'entrée pour l'agent avec l'historique COMPLET
    # L'agent a besoin de tout le contexte pour comprendre la dernière réponse
    messages_for_agent = list(chat_history.messages) + [last_user_message]
    
    # On convertit les messages au format attendu par client.beta.conversations.start
    role_map = {"human": "user", "ai": "assistant"}
    inputs_for_agent = [
        {"role": role_map.get(msg.type, "user"), "content": msg.content} 
        for msg in messages_for_agent
    ]

    try:
        resp = client.beta.conversations.start(
            agent_id=AGENT_ID_EXTRACTOR,
            inputs=inputs_for_agent, # Utiliser l'historique complet
            store=False
        )
        
        text_response = resp.outputs[0].content if resp.outputs else ""
        extracted_data = json.loads(text_response)
        print(f"   > Profile data extracted by Agent 1: {extracted_data}")
        
        # 5. Mettre à jour l'historique avec le dernier message utilisateur
        # L'IA n'a pas répondu directement, donc on ajoute juste le message utilisateur.
        chat_history.add_message(last_user_message)
        
        return {"user_profile": extracted_data}
        
    except Exception as e:
        print(f"   > Error calling Mistral Agent 1: {e}")
        return {"user_profile": {"missing_fields": ["all"]}}



def ask_clarification_question_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Ce nœud appelle l'Agent Mistral AGENT_ID_2 pour générer une question
    et la sauvegarde dans l'historique.
    """
    print("--- NODE: Clarification Question (calling Mistral Agent API) ---")
    
    # 1. Récupérer le session_id
    session_id = config["configurable"]["session_id"]
    chat_history = get_chat_history(session_id)
    
    profile = state.get('user_profile', {})
    
    input_for_agent_2 = {
        "language": profile.get('language'),
        "fields_to_ask": profile.get('missing_fields', []),
        "interests": profile.get('interests'),
        "constraints": profile.get('constraints'),
        "meals_services": profile.get('meals_services'),
        "tickets_logistics": profile.get('tickets_logistics'),
        "time_window": profile.get('time_window'),
    }
    input_str = json.dumps(input_for_agent_2)
    
    try:
        resp_2 = client.beta.conversations.start(
            agent_id=AGENT_ID_QUESTIONER,
            inputs=[{"role": "user", "content": input_str}],
            store=False
        )
        question_text = resp_2.outputs[0].content if resp_2.outputs else "Désolé, je ne sais pas quoi demander."
        
        # 2. Créer un message IA et le sauvegarder dans l'historique
        ai_question_message = AIMessage(content=question_text)
        chat_history.add_message(ai_question_message)
        
        print(f"   > Question generated by Agent 2: {question_text}")
        
        return {
            "messages": [ai_question_message],
            "question_attempts": state.get('question_attempts', 0) + 1
        }
    except Exception as e:
        # Gérer l'erreur et sauvegarder aussi le message d'erreur
        error_message = AIMessage(content="J'ai un problème. Pourriez-vous reformuler ?")
        chat_history.add_message(error_message)
        print(f"   > Error calling Mistral Agent 2: {e}")
        return {"messages": [error_message]}

# --- Agent Météo (Amélioré) ---
def weather_tool(state: GraphState) -> dict:
    print("--- Tool Call: Real Weather Agent ---")
    profile = state.get('user_profile', {})
    visit_date_str = profile.get('visit_date', datetime.now().strftime('%Y-%m-%d'))
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 48.8049, "longitude": 2.1204, "hourly": "temperature_2m,precipitation_probability", "timezone": "Europe/Paris", "start_date": visit_date_str, "end_date": visit_date_str}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        avg_temp = sum(data["hourly"]["temperature_2m"]) / len(data["hourly"]["temperature_2m"])
        max_precip_prob = max(data["hourly"]["precipitation_probability"])
        temp_desc = "cold" if avg_temp < 10 else "cool" if avg_temp < 15 else "pleasant" if avg_temp <= 25 else "warm"
        precip_desc = "high chance of rain" if max_precip_prob > 50 else "low chance of rain"
        summary = f"The weather will be generally {temp_desc} with a {precip_desc} (average temp: {avg_temp:.0f}°C)."
        return {"collected_data": {"weather_summary": summary}}
    except requests.exceptions.RequestException as e:
        return {"collected_data": {"weather_summary": f"Weather API error: {e}"}}
    
def rag_search_node(state: GraphState) -> dict:
    """Appelle l'agent RAG avec les intérêts filtrés de l'utilisateur."""
    print("--- NODE: RAG Search ---")
    
    profile = state.get('user_profile', {})
    interests_dict = profile.get('interests', {})
    
    # --- CORRECTION : Filtrer les intérêts pour ne garder que ceux > 0 ---
    # On transforme le dictionnaire d'intérêts en une liste de noms d'intérêts.
    active_interests = [interest for interest, score in interests_dict.items() if isinstance(score, (int, float)) and score > 0]
    
    # Si après filtrage il n'y a plus d'intérêts, utiliser une valeur par défaut.
    if not active_interests:
        print("   > No active interests found, using default query")
        active_interests = ["histoire générale", "architecture"] # Utilisez des termes plus génériques
    # --- FIN DE LA CORRECTION ---
    
    print(f"   > Searching for active interests: {active_interests}")
    
    # Appeler l'agent RAG avec la liste de chaînes de caractères
    rag_results = rag_agent.search_objects(interests=active_interests, k=5)
    
    print(f"   > Found {len(rag_results)} interest categories")
    for interest, objects in rag_results.items():
        print(f"     - {interest}: {len(objects)} objects")
    
    # Retourner les résultats pour la fusion
    return {"collected_data": {"rag_results": rag_results}}

def weather_node(state: GraphState) -> dict:
    """Appelle l'outil météo."""
    print("--- NODE: Weather ---")
    return weather_tool(state) # La fonction est déjà un "outil" qui prend l'état

# =============================================================================
# PLANNING AGENT - Complete Implementation
# =============================================================================

import json
from typing import Dict, List, Any
from collections import defaultdict



class PlanningAgent:
    """
    Takes RAG results and contextual data to create a prioritized itinerary.
    Output format includes priority scores at group, subgroup, and place levels.
    """
    
    def __init__(self, mistral_client=None, agent_id=None):
        """
        Args:
            mistral_client: Optional Mistral client for LLM-based planning
            agent_id: Optional Mistral agent ID for planning
        """
        self.mistral_client = mistral_client
        self.agent_id = agent_id
    
    def aggregate_rag_results(self, rag_results: Dict[str, List[Dict]]) -> Dict:
        """
        Aggregates RAG results to calculate priorities at each hierarchical level.
        
        Input format (from RAG agent):
        {
            "interest_1": [
                {
                    "group_id": "G1",
                    "group_title": "Palace",
                    "subgroup_id": "SG1",
                    "subgroup_title": "Royal Apartments",
                    "place_id": "P1",
                    "place_title": "King's Chamber",
                    "object_id": "O1",
                    "object_title": "Royal Bed",
                    "score": 0.85
                },
                ...
            ],
            "interest_2": [...]
        }
        
        Output format:
        {
            "groups": {
                "G1": {
                    "group_id": "G1",
                    "title": "Palace",
                    "priority": 0.92,
                    "object_count": 15,
                    "matched_interests": ["monarchy", "architecture"]
                }
            },
            "subgroups": {
                "SG1": {
                    "subgroup_id": "SG1",
                    "title": "Royal Apartments",
                    "group_id": "G1",
                    "priority": 0.88,
                    "object_count": 8,
                    "matched_interests": ["monarchy"]
                }
            },
            "places": {
                "P1": {
                    "place_id": "P1",
                    "title": "King's Chamber",
                    "subgroup_id": "SG1",
                    "group_id": "G1",
                    "priority": 0.85,
                    "object_count": 3,
                    "objects": [
                        {
                            "object_id": "O1",
                            "title": "Royal Bed",
                            "priority": 0.85,
                            "matched_interests": ["monarchy"]
                        }
                    ]
                }
            }
        }
        """
        
        # Initialize aggregation structures
        groups = defaultdict(lambda: {
            "object_count": 0,
            "total_score": 0.0,
            "scores": [],
            "matched_interests": set()
        })
        
        subgroups = defaultdict(lambda: {
            "object_count": 0,
            "total_score": 0.0,
            "scores": [],
            "matched_interests": set()
        })
        
        places = defaultdict(lambda: {
            "object_count": 0,
            "total_score": 0.0,
            "scores": [],
            "matched_interests": set(),
            "objects": []
        })
        
        # Process each interest and its matched objects
        for interest, objects in rag_results.items():
            for obj in objects:
                group_id = obj["group_id"]
                subgroup_id = obj["subgroup_id"]
                place_id = obj["place_id"]
                score = obj["score"]
                
                # Aggregate at group level
                if "title" not in groups[group_id]:
                    groups[group_id].update({
                        "group_id": group_id,
                        "title": obj["group_title"]
                    })
                groups[group_id]["object_count"] += 1
                groups[group_id]["total_score"] += score
                groups[group_id]["scores"].append(score)
                groups[group_id]["matched_interests"].add(interest)
                
                # Aggregate at subgroup level
                if "title" not in subgroups[subgroup_id]:
                    subgroups[subgroup_id].update({
                        "subgroup_id": subgroup_id,
                        "title": obj["subgroup_title"],
                        "group_id": group_id
                    })
                subgroups[subgroup_id]["object_count"] += 1
                subgroups[subgroup_id]["total_score"] += score
                subgroups[subgroup_id]["scores"].append(score)
                subgroups[subgroup_id]["matched_interests"].add(interest)
                
                # Aggregate at place level
                if "title" not in places[place_id]:
                    places[place_id].update({
                        "place_id": place_id,
                        "title": obj["place_title"],
                        "subgroup_id": subgroup_id,
                        "group_id": group_id
                    })
                places[place_id]["object_count"] += 1
                places[place_id]["total_score"] += score
                places[place_id]["scores"].append(score)
                places[place_id]["matched_interests"].add(interest)
                places[place_id]["objects"].append({
                    "object_id": obj["object_id"],
                    "title": obj["object_title"],
                    "priority": score,
                    "matched_interests": [interest]
                })
        
        # Calculate priorities (using weighted average of top scores)
        def calculate_priority(item_data):
            scores = sorted(item_data["scores"], reverse=True)
            if not scores:
                return 0.0
            # Weight: top score gets 50%, rest averaged for other 50%
            top_score = scores[0]
            avg_rest = sum(scores[1:]) / len(scores[1:]) if len(scores) > 1 else top_score
            return 0.5 * top_score + 0.5 * avg_rest
        
        # Finalize groups
        final_groups = {}
        for gid, gdata in groups.items():
            final_groups[gid] = {
                "group_id": gdata["group_id"],
                "title": gdata["title"],
                "priority": calculate_priority(gdata),
                "object_count": gdata["object_count"],
                "matched_interests": list(gdata["matched_interests"])
            }
        
        # Finalize subgroups
        final_subgroups = {}
        for sgid, sgdata in subgroups.items():
            final_subgroups[sgid] = {
                "subgroup_id": sgdata["subgroup_id"],
                "title": sgdata["title"],
                "group_id": sgdata["group_id"],
                "priority": calculate_priority(sgdata),
                "object_count": sgdata["object_count"],
                "matched_interests": list(sgdata["matched_interests"])
            }
        
        # Finalize places
        final_places = {}
        for pid, pdata in places.items():
            # Sort objects by priority
            sorted_objects = sorted(pdata["objects"], key=lambda x: x["priority"], reverse=True)
            final_places[pid] = {
                "place_id": pdata["place_id"],
                "title": pdata["title"],
                "subgroup_id": pdata["subgroup_id"],
                "group_id": pdata["group_id"],
                "priority": calculate_priority(pdata),
                "object_count": pdata["object_count"],
                "objects": sorted_objects,
                "matched_interests": list(pdata["matched_interests"])
            }
        
        return {
            "groups": final_groups,
            "subgroups": final_subgroups,
            "places": final_places
        }
    
    def create_time_based_itinerary(
        self, 
        aggregated_data: Dict,
        user_profile: Dict,
        weather_data: Dict
    ) -> Dict:
        """
        Creates a time-sequenced itinerary based on priorities and constraints.
        
        Returns:
        {
            "itinerary": [
                {
                    "time_slot": "10:00-11:30",
                    "group": {...},
                    "subgroup": {...},
                    "places": [
                        {
                            "place": {...},
                            "suggested_duration_minutes": 30,
                            "highlights": [top objects]
                        }
                    ]
                }
            ],
            "summary": {
                "total_duration_minutes": 240,
                "total_places": 8,
                "total_objects": 25
            }
        }
        """
        
        time_window = user_profile.get('time_window', {})
        
        # Utiliser une valeur par défaut si la clé est absente OU si la valeur est None
        start_time = time_window.get('start') if time_window.get('start') else '10:00'
        end_time = time_window.get('end') if time_window.get('end') else '16:00'
        
        # Extraire la durée et la convertir en minutes si disponible
        duration_hours = user_profile.get('duration_hours')
        if duration_hours:
            available_minutes = int(duration_hours) * 60
        else:
            # Calculer la durée si start/end sont fournis
            from datetime import datetime, timedelta
            start = datetime.strptime(start_time, '%H:%M')
            end = datetime.strptime(end_time, '%H:%M')
            available_minutes = int((end - start).total_seconds() / 60)
        
        # Get sorted groups by priority
        sorted_groups = sorted(
            aggregated_data["groups"].values(),
            key=lambda x: x["priority"],
            reverse=True
        )
        
        itinerary = []
        current_time = start
        remaining_minutes = available_minutes
        
        for group in sorted_groups:
            if remaining_minutes <= 30:  # Minimum 30 min per section
                break
            
            # Get subgroups for this group
            group_subgroups = [
                sg for sg in aggregated_data["subgroups"].values()
                if sg["group_id"] == group["group_id"]
            ]
            sorted_subgroups = sorted(group_subgroups, key=lambda x: x["priority"], reverse=True)
            
            for subgroup in sorted_subgroups[:2]:  # Top 2 subgroups per group
                # Get places for this subgroup
                subgroup_places = [
                    p for p in aggregated_data["places"].values()
                    if p["subgroup_id"] == subgroup["subgroup_id"]
                ]
                sorted_places = sorted(subgroup_places, key=lambda x: x["priority"], reverse=True)
                
                # Allocate time based on priority
                slot_duration = min(90, remaining_minutes)  # Max 90 min per slot
                places_in_slot = []
                
                for place in sorted_places[:3]:  # Top 3 places per subgroup
                    duration = 20 + (place["object_count"] * 5)  # Base + 5min per object
                    duration = min(duration, 45)  # Cap at 45 min per place
                    
                    places_in_slot.append({
                        "place": place,
                        "suggested_duration_minutes": duration,
                        "highlights": place["objects"][:3]  # Top 3 objects
                    })
                
                end_slot = current_time + timedelta(minutes=slot_duration)
                itinerary.append({
                    "time_slot": f"{current_time.strftime('%H:%M')}-{end_slot.strftime('%H:%M')}",
                    "group": group,
                    "subgroup": subgroup,
                    "places": places_in_slot
                })
                
                current_time = end_slot
                remaining_minutes -= slot_duration
                
                if remaining_minutes <= 30:
                    break
        
        # Calculate summary
        total_places = sum(len(slot["places"]) for slot in itinerary)
        total_objects = sum(
            len(place["highlights"])
            for slot in itinerary
            for place in slot["places"]
        )
        
        return {
            "itinerary": itinerary,
            "summary": {
                "total_duration_minutes": available_minutes - remaining_minutes,
                "total_places": total_places,
                "total_objects": total_objects,
                "weather_consideration": weather_data.get("weather_summary", "")
            },
            "metadata": {
                "user_interests": user_profile.get("interests", []),
                "constraints": user_profile.get("constraints", []),
                "language": user_profile.get("language", "en")
            }
        }
    
    def plan(
        self,
        rag_results: Dict,
        user_profile: Dict,
        weather_data: Dict
    ) -> Dict:
        """
        Main planning method. Orchestrates the full planning process.
        
        Args:
            rag_results: Output from RAG agent
            user_profile: User profile from extraction agent
            weather_data: Weather data from weather tool
            
        Returns:
            Complete itinerary with priorities at all levels
        """
        print("🗓️ Planning Agent: Starting itinerary creation...")
        
        # Step 1: Aggregate RAG results with priorities
        print("   > Aggregating locations by priority...")
        aggregated = self.aggregate_rag_results(rag_results)
        
        print(f"   > Found {len(aggregated['groups'])} groups, "
              f"{len(aggregated['subgroups'])} subgroups, "
              f"{len(aggregated['places'])} places")
        
        # Step 2: Create time-based itinerary
        print("   > Creating time-based itinerary...")
        itinerary = self.create_time_based_itinerary(
            aggregated,
            user_profile,
            weather_data
        )
        
        # Step 3: Add aggregated data to output
        itinerary["prioritized_locations"] = aggregated
        
        print(f"   > Itinerary created: {len(itinerary['itinerary'])} time slots")
        
        return itinerary


# =============================================================================
# VERSION 2 : AGENT DE PLANIFICATION BASÉ SUR UN LLM (LANGGRAPH DIRECT)
# =============================================================================

class LLMPlanningAgent:
    """
    Crée un itinéraire en appelant directement un modèle de langage (LLM)
    via l'API ChatMistralAI. Respecte le même format d'entrée/sortie que l'agent original.
    """
    
    def __init__(self, llm: Any):
        """
        Args:
            llm: Une instance initialisée d'un modèle de langage compatible LangChain (ex: ChatMistralAI).
        """
        if not llm:
            raise ValueError("An initialized LLM instance must be provided.")
        self.llm = llm

    # On réutilise cette méthode car elle est excellente pour préparer un contexte
    # synthétique et de haute qualité pour le LLM.
    def aggregate_rag_results(self, rag_results: Dict[str, List[Dict]]) -> Dict:
        # (Le code de cette fonction est identique à l'original. Je l'ai inclus pour que la classe soit complète.)
        groups = defaultdict(lambda: {"object_count": 0, "total_score": 0.0, "scores": [], "matched_interests": set()})
        subgroups = defaultdict(lambda: {"object_count": 0, "total_score": 0.0, "scores": [], "matched_interests": set()})
        places = defaultdict(lambda: {"object_count": 0, "total_score": 0.0, "scores": [], "matched_interests": set(), "objects": []})
        for interest, objects in rag_results.items():
            for obj in objects:
                gid, sgid, pid, score = obj["group_id"], obj["subgroup_id"], obj["place_id"], obj["score"]
                if "title" not in groups[gid]: groups[gid].update({"group_id": gid, "title": obj["group_title"]})
                groups[gid]["object_count"] += 1; groups[gid]["total_score"] += score; groups[gid]["scores"].append(score); groups[gid]["matched_interests"].add(interest)
                if "title" not in subgroups[sgid]: subgroups[sgid].update({"subgroup_id": sgid, "title": obj["subgroup_title"], "group_id": gid})
                subgroups[sgid]["object_count"] += 1; subgroups[sgid]["total_score"] += score; subgroups[sgid]["scores"].append(score); subgroups[sgid]["matched_interests"].add(interest)
                if "title" not in places[pid]: places[pid].update({"place_id": pid, "title": obj["place_title"], "subgroup_id": sgid, "group_id": gid})
                places[pid]["object_count"] += 1; places[pid]["total_score"] += score; places[pid]["scores"].append(score); places[pid]["matched_interests"].add(interest)
                places[pid]["objects"].append({"object_id": obj["object_id"], "title": obj["object_title"], "priority": score, "matched_interests": [interest]})
        def calculate_priority(data):
            scores = sorted(data["scores"], reverse=True)
            if not scores: return 0.0
            top_score, avg_rest = scores[0], sum(scores[1:]) / len(scores[1:]) if len(scores) > 1 else scores[0]
            return 0.5 * top_score + 0.5 * avg_rest
        final_groups = {gid: {"group_id": g["group_id"], "title": g["title"], "priority": calculate_priority(g), "object_count": g["object_count"], "matched_interests": list(g["matched_interests"])} for gid, g in groups.items()}
        final_subgroups = {sgid: {"subgroup_id": sg["subgroup_id"], "title": sg["title"], "group_id": sg["group_id"], "priority": calculate_priority(sg), "object_count": sg["object_count"], "matched_interests": list(sg["matched_interests"])} for sgid, sg in subgroups.items()}
        final_places = {pid: {"place_id": p["place_id"], "title": p["title"], "subgroup_id": p["subgroup_id"], "group_id": p["group_id"], "priority": calculate_priority(p), "object_count": p["object_count"], "objects": sorted(p["objects"], key=lambda x: x["priority"], reverse=True), "matched_interests": list(p["matched_interests"])} for pid, p in places.items()}
        return {"groups": final_groups, "subgroups": final_subgroups, "places": final_places}

    def _build_planning_prompt(self, aggregated_data: Dict, user_profile: Dict, weather_data: Dict) -> str:
        """Construit le prompt final à envoyer au LLM pour la planification."""
        
        # Le format de sortie JSON est la partie la plus importante du prompt.
        output_format_description = """
        {
            "itinerary": [
                {
                    "time_slot": "HH:MM-HH:MM",
                    "group": {"group_id": "...", "title": "..."},
                    "subgroup": {"subgroup_id": "...", "title": "..."},
                    "places": [
                        {
                            "place": {"place_id": "...", "title": "..."},
                            "suggested_duration_minutes": 30,
                            "highlights": [ {"title": "..."} ]
                        }
                    ]
                }
            ],
            "summary": {
                "total_duration_minutes": 240,
                "total_places": 8,
                "total_objects": 25,
                "weather_consideration": "Un commentaire sur l'impact de la météo."
            },
            "metadata": {
                "user_interests": ["liste", "des", "intérêts"],
                "constraints": ["liste", "des", "contraintes"],
                "language": "fr"
            }
        }
        """

        # On ne garde que les lieux les plus pertinents pour le prompt afin de le garder concis.
        top_places = sorted(aggregated_data["places"].values(), key=lambda p: p['priority'], reverse=True)[:10]

        prompt = f"""
        Vous êtes un expert planificateur de visites pour le Château de Versailles.
        Votre mission est de créer un itinéraire personnalisé, réaliste et agréable en vous basant sur les informations fournies.

        Voici les informations disponibles :

        1. PROFIL UTILISATEUR :
        {json.dumps(user_profile, indent=2, ensure_ascii=False)}

        2. MÉTÉO PRÉVUE :
        {json.dumps(weather_data, indent=2, ensure_ascii=False)}

        3. TOP 10 DES LIEUX RECOMMANDÉS (basés sur les intérêts de l'utilisateur) :
        {json.dumps({"top_places": top_places}, indent=2, ensure_ascii=False)}

        Instructions :
        1. Crée un itinéraire détaillé en sélectionnant les lieux les plus pertinents dans la liste fournie.
        2. Respecte la durée totale de la visite (champ 'duration_hours' ou la différence entre 'start' et 'end').
        3. Prends en compte les contraintes ('constraints'), le rythme ('pace') et la météo. Suggère des activités intérieures si la météo est mauvaise.
        4. Alloue des durées réalistes pour chaque lieu ('suggested_duration_minutes').
        5. Remplis la section 'summary' avec des totaux précis.

        Ta réponse doit être UNIQUEMENT un objet JSON valide, sans aucun texte, commentaire, ou formatage markdown avant ou après.
        Le JSON doit suivre STRICTEMENT la structure suivante :
        ---
        {output_format_description}
        ---
        """
        return prompt

    def _parse_llm_response(self, response_content: str) -> Dict:
        """Extrait et parse le bloc JSON de la réponse du LLM de manière robuste."""
        try:
            # Essayer de parser directement, au cas où le LLM respecte parfaitement les consignes.
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Si ça échoue, utiliser une expression régulière pour trouver un objet JSON
            # (souvent enrobé dans ```json ... ``` ou du texte).
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"   > Error: LLM response contained invalid JSON after regex extraction: {e}")
                    return {"error": f"LLM produced invalid JSON: {e}"}
            else:
                print("   > Error: No JSON object found in the LLM response.")
                print(f"   > Response was: {response_content}")
                return {"error": "Failed to generate a valid itinerary from LLM response."}

    def plan(self, rag_results: Dict, user_profile: Dict, weather_data: Dict) -> Dict:
        """
        Méthode de planification principale qui orchestre le processus avec le LLM.
        """
        print("🗓️ LLM Planning Agent: Starting itinerary creation...")
        
        print("   > Aggregating RAG results for LLM context...")
        aggregated_data = self.aggregate_rag_results(rag_results)
        
        print("   > Building prompt for LLM planner...")
        prompt = self._build_planning_prompt(aggregated_data, user_profile, weather_data)
        
        print("   > Sending planning prompt to LLM...")
        try:
            response = self.llm.invoke(prompt)
            itinerary = self._parse_llm_response(response.content)
        except Exception as e:
            print(f"   > Unexpected error during LLM invocation: {e}")
            itinerary = {"error": f"LLM invocation failed: {e}"}

        # Pour respecter le format de sortie, on ajoute les données agrégées.
        itinerary["prioritized_locations"] = aggregated_data
        
        print("   > LLM-based itinerary created.")
        return itinerary

# Initialize planning agent 
planning_agent = PlanningAgent()
# Nouvel agent de planification (basé sur le LLM)
llm_planning_agent = LLMPlanningAgent(llm=llm_planner_model)

def planning_node(state: GraphState) -> dict:
    """
    Nœud LangGraph qui appelle l'agent de planification.
    """
    print("--- NODE: Planning (creating itinerary) ---")
    
    collected_data = state.get('collected_data', {})
    user_profile = state.get('user_profile', {})
    rag_results = collected_data.get('rag_results', {})
    weather_data = {"weather_summary": collected_data.get('weather_summary', {})}
    
    if not rag_results:
        # ... (gestion de l'erreur si pas de résultats RAG)
        return {"collected_data": {"itinerary": {"error": "No locations found."}}}
    
    try:
        # --- CHOISISSEZ VOTRE AGENT ICI ---
        
        # Option 1: Utiliser l'agent basé sur des règles (votre original)
        # print("   > Using Rule-Based Planning Agent")
        # itinerary = rule_based_planning_agent.plan(
        #     rag_results=rag_results,
        #     user_profile=user_profile,
        #     weather_data=weather_data
        # )

        # Option 2: Utiliser le nouvel agent basé sur le LLM
        print("   > Using LLM-Based Planning Agent")
        itinerary = llm_planning_agent.plan(
            rag_results=rag_results,
            user_profile=user_profile,
            weather_data=weather_data
        )
        # --- FIN DU CHOIX ---
        
        updated_collected_data = collected_data.copy()
        updated_collected_data['itinerary'] = itinerary
        
        return {"collected_data": updated_collected_data}
        
    except Exception as e:
        print(f"   > Error in planning: {e}")
        import traceback
        traceback.print_exc()
        return {"collected_data": {"itinerary": {"error": f"Planning failed: {str(e)}"}}}


def synthesis_node(state: GraphState) -> dict:
    """Nœud final qui rassemble les données (pour l'instant, il ne fait que les afficher)."""
    print("--- NODE: Synthesis ---")
    print("Données collectées avant la synthèse finale :")
    print(json.dumps(state.get('collected_data'), indent=2, ensure_ascii=False))
    # Plus tard, ce nœud appellera le formateur de réponse
    return {}

# =============================================================================
# ÉTAPE 4 : ROUTAGE ET ASSEMBLAGE
# =============================================================================

from langgraph.types import Send

ONE_SHOT = True
def decide_next_step(state: GraphState, config: dict | None = None):
    """Route toujours vers la collecte si ONE_SHOT est actif."""
    print("--- ROUTER: Deciding Next Step ---")
    cfg = (config or {}).get("configurable", {})
    force_one_shot = cfg.get("one_shot", False) or ONE_SHOT

    if force_one_shot:
        print("   > ONE_SHOT actif. Fan-out direct vers rag_search_node + weather_node.")
        return [
            Send("rag_search_node", state),
            Send("weather_node", state)
        ]

    if state.get('user_profile', {}).get('missing_fields') and state.get('question_attempts', 0) < 3:
        print("   > Profile incomplet. Clarification.")
        return "ask_clarification_question"
    else:
        print("   > Profile complet. Fan-out gather.")
        return [
            Send("rag_search_node", state),
            Send("weather_node", state)
        ]





# Update your graph builder:
builder = StateGraph(GraphState)

# Phase 1

builder.add_node("information_extractor", information_extractor_node)
builder.add_node("ask_clarification_question", ask_clarification_question_node)

# Phase 2 (Nœuds parallèles)
builder.add_node("rag_search_node", rag_search_node)
builder.add_node("weather_node", weather_node)

# Phase 3 (Planning)
builder.add_node("planning_node", planning_node)

# Phase 4 (Synthesis)
builder.add_node("synthesis_node", synthesis_node)

# Entry point
builder.set_entry_point("information_extractor")
#builder.add_edge(START, information_extractor_node)

# Conditional routing from extractor
builder.add_conditional_edges(
    "information_extractor",
    decide_next_step,
    # No path_map needed when using Send
)

# Loop back for clarificationx
#builder.add_edge("ask_clarification_question", "information_extractor")

# After parallel nodes, converge to planning
builder.add_edge("rag_search_node", "planning_node")
builder.add_edge("weather_node", "planning_node")

# After planning, go to synthesis
builder.add_edge("planning_node", "synthesis_node")

# End
builder.add_edge("synthesis_node", END)
 
#checkpointer = InMemorySaver()
le_guide_royal = builder.compile()
print("\n--- Graph with Data Gathering Phase Compiled Successfully ---")


# from IPython.display import Image, display

# display(Image(le_guide_royal.get_graph().draw_mermaid_png()))

def format_itinerary_to_text(itinerary: dict) -> str:
    """Transforme l'itinéraire (dict) en réponse naturelle en français."""
    if not itinerary or "itinerary" not in itinerary:
        return "Voici quelques recommandations de visite à Versailles en fonction de vos intérêts."

    slots = itinerary.get("itinerary", [])
    summary = itinerary.get("summary", {})

    lines = []
    if summary:
        total = summary.get("total_duration_minutes")
        weather = summary.get("weather_consideration", "")
        if total:
            lines.append(f"💡 Itinéraire proposé (~{total} minutes).")
        if weather:
            lines.append(f"🌦️ Météo : {weather}")

    for slot in slots:
        time_slot = slot.get("time_slot", "")
        group = slot.get("group", {}).get("title", "")
        subgroup = slot.get("subgroup", {}).get("title", "")
        lines.append(f"\n⏱️ {time_slot} — {group} › {subgroup}")

        for p in slot.get("places", []):
            place = p.get("place", {}).get("title", "Lieu")
            dur = p.get("suggested_duration_minutes", 30)
            highlights = p.get("highlights", [])
            hl = ", ".join(o.get("title", "") for o in highlights[:3]) if highlights else ""
            if hl:
                lines.append(f"  • {place} (~{dur} min) — à voir : {hl}")
            else:
                lines.append(f"  • {place} (~{dur} min)")

    if not lines:
        return "Je vous recommande de commencer par les Grands Appartements, puis la Galerie des Glaces, et de terminer par les Jardins si le temps le permet."

    return "\n".join(lines)



# ======================= API FASTAPI (À COLLER EN BAS) =======================
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# ---- App FastAPI ----
app = FastAPI(title="Versailles Planner API", version="2.0.0")

# CORS large pour tests – à restreindre en prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # remplace par tes domaines front en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Schémas E/S ----
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message utilisateur")
    session_id: Optional[str] = Field(None, description="Identifiant de session (conserve le contexte)")

class ChatResponse(BaseModel):
    session_id: str
    user_profile: Dict[str, Any] = {}
    collected_data: Dict[str, Any] = {}
    assistant_message: Optional[str] = None

class SearchRAGRequest(BaseModel):
    interests: List[str] = Field(..., min_items=1)
    k: int = Field(5, ge=1, le=50)

class ResetSessionRequest(BaseModel):
    session_id: str

# ---- Endpoints ----
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# ---- Nouveau schéma pour coller aux consignes d’évaluation ----
class EvalChatRequest(BaseModel):
    question: str = Field(..., description="Question du visiteur")

class EvalChatResponse(BaseModel):
    answer: str = Field(..., description="Réponse complète du chatbot")

@app.post("/chat", response_model=EvalChatResponse)
def chat(req: EvalChatRequest):
    if "le_guide_royal" not in globals():
        raise HTTPException(status_code=500, detail="Graph not compiled")

    session_id = str(uuid.uuid4())
    initial_state = {
        "messages": [HumanMessage(content=req.question)],
        "user_profile": {},
        "question_attempts": 0,
        "collected_data": {},
    }
    # 👉 on force le one-shot aussi au runtime (en plus du flag global)
    run_config = {"configurable": {"session_id": session_id, "one_shot": True}}

    try:
        final_state = le_guide_royal.invoke(initial_state, config=run_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")

    # 1) on ignore volontairement les clarifications (one-shot)
    # 2) on formate un texte de réponse à partir de l'itinéraire
    itinerary = final_state.get("collected_data", {}).get("itinerary")
    if itinerary:
        answer_text = format_itinerary_to_text(itinerary)
    else:
        # fallback : dernier message IA si vraiment rien
        msgs = final_state.get("messages", [])
        last_ai = next((m for m in reversed(msgs) if getattr(m, "type", None) == "ai"), None)
        answer_text = last_ai.content if last_ai else "Voici une proposition de parcours à Versailles : commencez par les Grands Appartements, puis la Galerie des Glaces, et terminez par les Jardins si la météo le permet."

    return {"answer": answer_text}

