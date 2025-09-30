# streamlit_app.py
"""
Application Streamlit complète pour le Guide Royal de Versailles
Intègre la transcription vocale et l'orchestration complète du workflow LangGraph
"""

import streamlit as st
import io
import os
import json
import base64
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Imports pour la transcription vocale
try:
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.warning("pydub n'est pas installé. Mode vocal désactivé.")

# Imports du workflow
try:
    from main import (
        le_guide_royal,
        get_chat_history,
        MISTRAL_API_KEY
    )
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    st.error(f"Erreur d'import du workflow: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Guide Royal Versailles",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .itinerary-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS DE TRANSCRIPTION VOCALE
# ============================================================================

def convert_to_wav_16k_mono(audio_bytes: bytes) -> Optional[bytes]:
    """Convertit l'audio en WAV 16kHz mono pour l'API Mistral."""
    if not AUDIO_AVAILABLE:
        return None
    
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)
        out = io.BytesIO()
        audio.export(out, format="wav")
        return out.getvalue()
    except Exception as e:
        st.error(f"Erreur de conversion audio: {e}")
        return None

def save_audio_locally(wav_bytes: bytes) -> Optional[Path]:
    """Sauvegarde l'audio localement pour debug."""
    try:
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = recordings_dir / f"recording_{timestamp}.wav"
        
        with open(filepath, "wb") as f:
            f.write(wav_bytes)
        
        return filepath
    except Exception as e:
        st.warning(f"Impossible de sauvegarder l'audio: {e}")
        return None

def transcribe_audio_mistral(audio_bytes: bytes, api_key: str) -> Dict[str, Any]:
    """
    Transcrit l'audio en utilisant l'API Mistral Voxtral.
    
    Returns:
        dict: {"success": bool, "text": str, "error": str}
    """
    try:
        import httpx
        from mistralai import Mistral
        
        # Conversion audio
        wav_bytes = convert_to_wav_16k_mono(audio_bytes)
        if not wav_bytes:
            return {"success": False, "error": "Échec de conversion audio"}
        
        # Sauvegarde locale (optionnel)
        saved_path = save_audio_locally(wav_bytes)
        
        # Encode en base64
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        
        # Appel API
        client = Mistral(api_key=api_key, client=httpx.Client(timeout=60))
        
        response = client.chat.complete(
            model="voxtral-mini-latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": audio_b64},
                    {"type": "text", "text": "Transcris cet audio en français et renvoie uniquement le texte brut sans formatage."}
                ]
            }]
        )
        
        # Extraction du texte
        text = response.choices[0].message.content.strip()
        
        return {
            "success": True,
            "text": text,
            "saved_path": str(saved_path) if saved_path else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Erreur API Mistral: {str(e)}",
            "traceback": traceback.format_exc()
        }

# ============================================================================
# FONCTIONS D'ORCHESTRATION DU WORKFLOW
# ============================================================================

def execute_workflow(user_input: str, session_id: str) -> Dict[str, Any]:
    """
    Exécute le workflow LangGraph avec l'entrée utilisateur.
    
    Args:
        user_input: Le texte de l'utilisateur (transcrit ou tapé)
        session_id: ID de session pour maintenir le contexte
    
    Returns:
        dict: Résultat du workflow avec les états intermédiaires
    """
    try:
        # Configuration avec session_id
        config = {
            "configurable": {
                "session_id": session_id,
                "thread_id": session_id
            }
        }
        
        # État initial
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_profile": {},
            "question_attempts": 0,
            "collected_data": {}
        }
        
        # Exécution du workflow avec streaming
        events = []
        final_state = None
        
        for event in le_guide_royal.stream(initial_state, config, stream_mode="values"):
            events.append(event)
            final_state = event
        
        return {
            "success": True,
            "state": final_state,
            "events": events
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def get_conversation_phase(state: Dict[str, Any]) -> str:
    """
    Détermine la phase de la conversation en fonction de l'état.
    
    Returns:
        "initial" | "clarification" | "processing" | "completed"
    """
    if not state:
        return "initial"
    
    # Vérifier si on a un itinéraire final
    collected_data = state.get("collected_data", {})
    if collected_data.get("itinerary"):
        return "completed"
    
    # Vérifier si on attend une clarification
    messages = state.get("messages", [])
    if messages and len(messages) > 0:
        last_msg = messages[-1]
        if hasattr(last_msg, 'type') and last_msg.type == "ai":
            return "clarification"
    
    # Vérifier si on a des données RAG/Weather mais pas encore d'itinéraire
    if collected_data.get("rag_results") or collected_data.get("weather_summary"):
        return "processing"
    
    return "initial"

# ============================================================================
# INITIALISATION DE LA SESSION STREAMLIT
# ============================================================================

def init_session_state():
    """Initialise les variables de session Streamlit."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.datetime.now().timestamp()}"
    
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = None
    
    if 'conversation_phase' not in st.session_state:
        st.session_state.conversation_phase = "initial"
    
    if 'messages_display' not in st.session_state:
        st.session_state.messages_display = []

init_session_state()

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">👑 Guide Royal - Château de Versailles</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Information")
        
        st.markdown(f"""
        **Session ID:** `{st.session_state.session_id[:8]}...`
        
        **Phase:** `{st.session_state.conversation_phase}`
        """)
        
        if st.button("🔄 Nouvelle session"):
            st.session_state.session_id = f"session_{datetime.datetime.now().timestamp()}"
            st.session_state.workflow_state = None
            st.session_state.conversation_phase = "initial"
            st.session_state.messages_display = []
            st.rerun()
        
        st.divider()
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"Workflow disponible: {WORKFLOW_AVAILABLE}")
            st.write(f"Audio disponible: {AUDIO_AVAILABLE}")
            if st.session_state.workflow_state:
                st.json(st.session_state.workflow_state.get("user_profile", {}))
    
    # Zone de conversation
    if st.session_state.conversation_phase != "completed":
        display_conversation_interface()
    else:
        display_itinerary_result()

# ============================================================================
# INTERFACE DE CONVERSATION
# ============================================================================

def display_conversation_interface():
    """Affiche l'interface de conversation (initial ou clarification)."""
    
    # Affichage des messages précédents
    if st.session_state.workflow_state:
        attempts = st.session_state.workflow_state.get('question_attempts', 0)
        if attempts > 0:
            st.caption(f"💬 Question {attempts}/3")
    
    if st.session_state.messages_display:
        st.markdown("### 💬 Conversation")
        for msg in st.session_state.messages_display:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.markdown(f'<div class="chat-message user-message">👤 Vous: {content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🤖 Assistant: {content}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Titre selon la phase
    if st.session_state.conversation_phase == "initial":
        st.markdown("### 🎤 Décrivez votre visite")
        st.caption("Exemple: *Je veux visiter le château demain avec mes deux enfants. J'aimerais voir en particulier le Jardin et des lieux insolites.*")
    else:
        st.markdown("### ❓ Répondez à la question")
    
    # Choix du mode d'entrée
    col1, col2 = st.columns([1, 5])
    
    with col1:
        input_mode = st.radio(
            "Mode",
            ["💬 Texte", "🎤 Vocal"],
            label_visibility="collapsed",
            key=f"mode_{st.session_state.conversation_phase}_{len(st.session_state.messages_display)}"
        )
    
    user_input = None
    
    # ===== MODE VOCAL avec st.audio_input =====
    if input_mode == "🎤 Vocal":
        if not AUDIO_AVAILABLE:
            st.error("Le mode vocal nécessite pydub. Installez-le avec: pip install pydub")
        else:
            # Keys uniques basées sur la phase
            audio_key = f"audio_input_{st.session_state.conversation_phase}_{len(st.session_state.messages_display)}"
            hash_key = f"audio_hash_{audio_key}"
            
            st.write("### 🎙️ Enregistrement")
            st.write("Appuyez sur **Record** puis **Stop**. La transcription se fera automatiquement.")
            
            try:
                audio_obj = st.audio_input("Enregistrez votre message", key=audio_key)
            except Exception as e:
                st.error(f"st.audio_input non disponible: {e}")
                st.info("Utilisez Streamlit >= 1.30.0 pour l'enregistrement audio")
                audio_obj = None
            
            # Traitement de l'audio
            if audio_obj:
                try:
                    # Lire les bytes
                    if hasattr(audio_obj, "read"):
                        raw_bytes = audio_obj.read()
                    elif isinstance(audio_obj, (bytes, bytearray)):
                        raw_bytes = bytes(audio_obj)
                    else:
                        raw_bytes = audio_obj.getvalue()
                except Exception as e:
                    st.error(f"Impossible de lire l'enregistrement: {e}")
                    raw_bytes = None
                
                if raw_bytes:
                    # Hash pour éviter le double traitement
                    import hashlib
                    audio_hash = hashlib.sha256(raw_bytes).hexdigest()
                    
                    # Vérifier si déjà traité
                    if st.session_state.get(hash_key) == audio_hash:
                        st.info("⏸️ Cet enregistrement a déjà été traité.")
                    else:
                        # Marquer comme traité
                        st.session_state[hash_key] = audio_hash
                        
                        # Transcription
                        with st.spinner("🎧 Transcription en cours..."):
                            result = transcribe_audio_mistral(raw_bytes, MISTRAL_API_KEY)
                            
                            if result["success"]:
                                user_input = result["text"]
                                st.success(f"✅ Transcrit: *{user_input}*")
                                
                                if result.get("saved_path"):
                                    st.caption(f"📁 Sauvegardé: {result['saved_path']}")
                                
                                # Nettoyer le widget pour permettre un nouvel enregistrement
                                try:
                                    del st.session_state[audio_key]
                                    del st.session_state[hash_key]
                                except:
                                    pass
                                
                                # Traiter l'input
                                process_user_input(user_input)
                                
                            else:
                                st.error(f"❌ Erreur de transcription: {result['error']}")
                                with st.expander("Détails de l'erreur"):
                                    st.code(result.get("traceback", "Pas de traceback"))
                                
                                # Permettre un nouvel essai
                                try:
                                    del st.session_state[audio_key]
                                    del st.session_state[hash_key]
                                except:
                                    pass
    # ===== MODE TEXTE =====
    else:
        text_key = f"text_{st.session_state.conversation_phase}_{len(st.session_state.messages_display)}"
        
        user_input = st.text_area(
            "Votre message",
            placeholder="Tapez votre message ici...",
            height=100,
            label_visibility="collapsed",
            key=text_key
        )
        
        if st.button("📤 Envoyer", type="primary", key=f"send_{text_key}"):
            if user_input and user_input.strip():
                process_user_input(user_input.strip())
            else:
                st.warning("⚠️ Veuillez saisir un message")

# ============================================================================
# TRAITEMENT DE L'ENTRÉE UTILISATEUR
# ============================================================================

def process_user_input(user_input: str):
    """Traite l'entrée utilisateur et met à jour le workflow."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING USER INPUT: {user_input[:100]}...")
    print(f"Current phase: {st.session_state.conversation_phase}")
    print(f"Messages count: {len(st.session_state.messages_display)}")
    print(f"{'='*60}\n")
    
    # Ajouter le message à l'affichage
    st.session_state.messages_display.append({
        "role": "user",
        "content": user_input
    })
    
    # Exécuter le workflow
    with st.spinner("🤔 Analyse en cours..."):
        try:
            result = execute_workflow(user_input, st.session_state.session_id)
            
            if not result["success"]:
                st.error(f"❌ Erreur workflow: {result['error']}")
                with st.expander("Détails de l'erreur"):
                    st.code(result.get("traceback", "Pas de traceback"))
                return
            
            # Mettre à jour l'état
            st.session_state.workflow_state = result["state"]
            
            # Debug
            print(f"\nWorkflow result:")
            print(f"  - question_attempts: {result['state'].get('question_attempts', 0)}")
            print(f"  - missing_fields: {result['state'].get('user_profile', {}).get('missing_fields', [])}")
            
            # Déterminer la phase
            new_phase = get_conversation_phase(result["state"])
            st.session_state.conversation_phase = new_phase
            
            print(f"  - new phase: {new_phase}\n")
            
            # Si on a un message de l'assistant, l'ajouter
            messages = result["state"].get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'type') and last_msg.type == "ai":
                    st.session_state.messages_display.append({
                        "role": "assistant",
                        "content": last_msg.content
                    })
        
        except Exception as e:
            st.error(f"❌ Exception: {e}")
            import traceback
            with st.expander("Traceback complet"):
                st.code(traceback.format_exc())
            return
    
    st.rerun()
# ============================================================================
# AFFICHAGE DE L'ITINÉRAIRE FINAL
# ============================================================================

def display_itinerary_result():
    """Affiche l'itinéraire final ET le récit narratif."""
    
    st.success("✅ Votre guide personnalisé est prêt!")
    
    state = st.session_state.workflow_state
    collected_data = state.get("collected_data", {})
    
    # Vérifier si on a un récit narratif
    has_narrative = collected_data.get("narrative") and not collected_data.get("narrative", {}).get("error")
    
    # Créer des onglets pour séparer l'itinéraire technique et le récit
    if has_narrative:
        tab1, tab2, tab3 = st.tabs(["📖 Récit de votre visite", "🗺️ Itinéraire détaillé", "📊 Données techniques"])
        
        with tab1:
            # Afficher le récit narratif en premier (expérience utilisateur)
            display_narrative_guide(state)
        
        with tab2:
            # Afficher l'itinéraire technique
            display_technical_itinerary(state)
        
        with tab3:
            # Afficher les données brutes
            display_technical_data(state)
    
    else:
        # Si pas de récit, afficher seulement l'itinéraire
        tab1, tab2 = st.tabs(["🗺️ Itinéraire détaillé", "📊 Données techniques"])
        
        with tab1:
            display_technical_itinerary(state)
        
        with tab2:
            display_technical_data(state)
    
    # Actions globales
    st.divider()
    if st.button("🔄 Planifier une nouvelle visite", type="primary"):
        st.session_state.session_id = f"session_{datetime.datetime.now().timestamp()}"
        st.session_state.workflow_state = None
        st.session_state.conversation_phase = "initial"
        st.session_state.messages_display = []
        st.rerun()

def display_technical_itinerary(state: Dict[str, Any]):
    """Affiche l'itinéraire technique avec les créneaux horaires."""
    
    collected_data = state.get("collected_data", {})
    itinerary = collected_data.get("itinerary", {})
    
    if not itinerary or itinerary.get("error"):
        st.error("Une erreur s'est produite lors de la génération de l'itinéraire.")
        st.json(itinerary)
        return
    
    # Résumé
    st.markdown("## 📊 Résumé de votre visite")
    summary = itinerary.get("summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⏱️ Durée", f"{summary.get('total_duration_minutes', 0)} min")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📍 Lieux", summary.get('total_places', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎨 Objets", summary.get('total_objects', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        weather = summary.get('weather_consideration', '')
        st.metric("🌡️ Météo", "☀️" if "pleasant" in weather or "sun" in weather else "🌧️")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Météo détaillée
    if summary.get('weather_consideration'):
        st.info(f"☀️ **Météo:** {summary['weather_consideration']}")
    
    st.divider()
    
    # Itinéraire détaillé (code existant)
    st.markdown("## 🗺️ Planning horaire")
    
    itinerary_slots = itinerary.get("itinerary", [])
    
    for i, slot in enumerate(itinerary_slots, 1):
        with st.expander(
            f"**Créneau {i}: {slot['time_slot']}** - {slot['group']['title']} / {slot['subgroup']['title']}",
            expanded=(i == 1)
        ):
            # Affichage du groupe
            group_title = slot['group']['title']
            group_priority = slot['group'].get('priority')
            if group_priority is not None:
                st.markdown(f"**Groupe:** {group_title} (Priorité: {group_priority:.2f})")
            else:
                st.markdown(f"**Groupe:** {group_title}")
            
            # Affichage du sous-groupe
            subgroup_title = slot['subgroup']['title']
            subgroup_priority = slot['subgroup'].get('priority')
            if subgroup_priority is not None:
                st.markdown(f"**Sous-groupe:** {subgroup_title} (Priorité: {subgroup_priority:.2f})")
            else:
                st.markdown(f"**Sous-groupe:** {subgroup_title}")
            
            st.markdown("---")
            
            for place_info in slot['places']:
                place = place_info['place']
                duration = place_info['suggested_duration_minutes']
                
                st.markdown(f"### 📍 {place['title']}")
                st.markdown(f"*⏱️ Durée suggérée: **{duration} minutes***")
                
                # Priorité du lieu
                place_priority = place.get('priority')
                if place_priority is not None:
                    st.markdown(f"*🎯 Priorité: **{place_priority:.2f}***")
                
                # Highlights
                if place_info.get('highlights'):
                    st.markdown("**🌟 À ne pas manquer:**")
                    for obj in place_info['highlights']:
                        if isinstance(obj, dict):
                            obj_title = obj.get('title', 'Objet')
                            obj_priority = obj.get('priority')
                            if obj_priority is not None:
                                st.markdown(f"- **{obj_title}** (priorité: {obj_priority:.2f})")
                            else:
                                st.markdown(f"- {obj_title}")
                        else:
                            st.markdown(f"- {obj}")
                
                # Intérêts correspondants
                if place.get('matched_interests'):
                    interests_list = place['matched_interests']
                    if interests_list:
                        st.markdown(f"*💡 Correspond à vos intérêts: {', '.join(interests_list)}*")
                
                st.markdown("---")

def display_technical_data(state: Dict[str, Any]):
    """Affiche les données techniques brutes."""
    
    collected_data = state.get("collected_data", {})
    itinerary = collected_data.get("itinerary", {})
    
    # Métadonnées du profil
    with st.expander("📋 Profil utilisateur"):
        metadata = itinerary.get("metadata", {})
        st.json(metadata)
    
    # Lieux prioritaires
    with st.expander("🎯 Lieux prioritaires (scores)"):
        prioritized = itinerary.get("prioritized_locations", {})
        if prioritized:
            st.json(prioritized.get("places", {}))
        else:
            st.info("Pas de données de priorité disponibles")
    
    # Données RAG
    with st.expander("🔍 Résultats de recherche RAG"):
        rag_results = collected_data.get("rag_results", {})
        st.json(rag_results)
    
    # Données météo
    with st.expander("🌤️ Données météo"):
        st.write(collected_data.get("weather_summary", "Pas de données météo"))
    
    # JSON complet
    with st.expander("💾 Export JSON complet"):
        full_data = json.dumps(collected_data, indent=2, ensure_ascii=False)
        st.download_button(
            "Télécharger toutes les données",
            data=full_data,
            file_name=f"versailles_full_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        st.json(collected_data)

def display_narrative_guide(state: Dict[str, Any]):
    """Affiche le guide narratif généré par l'agent narrateur."""
    
    collected_data = state.get("collected_data", {})
    narrative = collected_data.get("narrative", {})
    
    if not narrative or narrative.get("error"):
        st.warning("Le récit narratif n'a pas pu être généré.")
        if narrative.get("error"):
            with st.expander("Détails de l'erreur"):
                st.error(narrative["error"])
                if narrative.get("raw_response"):
                    st.code(narrative["raw_response"])
        return
    
    # Titre principal
    st.markdown(f"## 📖 {narrative.get('title', 'Votre parcours narratif')}")
    
    # Vue d'ensemble des accès
    if narrative.get('accessOverview'):
        st.markdown("### 🗺️ Parcours")
        access_path = " → ".join(narrative['accessOverview'])
        st.info(f"**Chemin:** {access_path}")
    
    # Durée
    duration = narrative.get('itineraryDurationMinutes')
    if duration:
        st.markdown(f"**⏱️ Durée estimée:** {duration} minutes ({duration // 60}h{duration % 60:02d})")
    
    st.divider()
    
    # Étapes narratives
    steps = narrative.get('steps', [])
    
    if not steps:
        st.warning("Aucune étape narrative disponible.")
        return
    
    st.markdown("### 📚 Votre visite étape par étape")
    
    for i, step_data in enumerate(steps, 1):
        step_id = step_data.get('step', f'step{i}')
        step_text = step_data.get('text', 'Aucun texte disponible.')
        
        # Afficher l'étape dans un conteneur stylisé
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border-left: 4px solid #9c27b0;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
            ">
                <h4 style="color: #9c27b0; margin-top: 0;">Étape {i}</h4>
                <p style="line-height: 1.8; text-align: justify; margin-bottom: 0;">
                    {step_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        # Export du récit en JSON
        narrative_json = json.dumps(narrative, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Télécharger le récit (JSON)",
            data=narrative_json,
            file_name=f"recit_versailles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export en texte simple
        text_export = f"{narrative.get('title', 'Visite de Versailles')}\n\n"
        if narrative.get('accessOverview'):
            text_export += f"Parcours: {' → '.join(narrative['accessOverview'])}\n\n"
        for i, step_data in enumerate(steps, 1):
            text_export += f"Étape {i}:\n{step_data.get('text', '')}\n\n"
        
        st.download_button(
            "📄 Télécharger le récit (TXT)",
            data=text_export,
            file_name=f"recit_versailles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    if not WORKFLOW_AVAILABLE:
        st.error("Le workflow n'est pas disponible. Vérifiez que main.py est accessible.")
        st.stop()
    
    if not MISTRAL_API_KEY:
        st.error("MISTRAL_API_KEY n'est pas définie. Configurez votre fichier .env")
        st.stop()
    
    main()