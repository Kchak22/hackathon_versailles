import streamlit as st
import json
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Assurez-vous que le fichier 'main.py' a bien été corrigé 
# (suppression de l'arête de boucle, du checkpointer, et ajout du mappage des rôles).
from main import le_guide_royal

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Le Guide Royal - Versailles",
    page_icon="🏰",
    layout="wide"
)

# --- Initialisation de l'État de la Session Streamlit ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.config = {"configurable": {"session_id": st.session_state.session_id}}
    st.session_state.messages_for_display = []
    st.session_state.current_graph_state = {}
    st.session_state.phase = "greeting"
    # CORRECTION: Gérer le compteur de tentatives et sa limite dans l'état de Streamlit.
    st.session_state.question_attempts = 0
    st.session_state.MAX_ATTEMPTS = 3 # Vous pouvez ajuster cette limite

# --- CSS et En-tête (inchangés) ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Votre CSS ici
st.markdown("""<div class="main-header">...</div>""", unsafe_allow_html=True) # Votre en-tête ici

# --- Fonctions d'Aide ---

@st.cache_resource
def get_graph():
    """Charge et compile le graphe une seule fois pour la performance."""
    return le_guide_royal

graph = get_graph()

def invoke_graph(user_message: HumanMessage, attempts: int):
    """
    CORRECTION: La fonction passe maintenant le compteur de tentatives au graphe.
    L'état du graphe est réinitialisé à chaque appel, sauf pour l'historique de chat (géré par session_id).
    """
    # L'état initial pour cette invocation ne contient que le nouveau message
    # et le nombre de tentatives actuel.
    inputs = {
        "messages": [user_message],
        "question_attempts": attempts 
    }
    try:
        result = graph.invoke(inputs, st.session_state.config)
        return result
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'invocation du graphe : {str(e)}")
        return None

def display_chat_history():
    """Affiche l'historique des messages stockés pour l'affichage."""
    for msg in st.session_state.messages_for_display:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            st.markdown(f'<div class="chat-message assistant-message">🤖 <b>Guide:</b> {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message user-message">👤 <b>Vous:</b> {content}</div>', unsafe_allow_html=True)

# ... Les autres fonctions display_profile, display_itinerary, etc. restent inchangées ...
# (Assurez-vous qu'elles sont bien présentes dans votre code)
def display_profile(profile):
    """Display the user profile in a nice format"""
    st.markdown('<div class="profile-box">', unsafe_allow_html=True)
    st.markdown("### 📋 Your Visit Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if profile.get('visit_date'):
            st.write(f"**Visit Date:** {profile['visit_date']}")
        if profile.get('time_window'):
            tw = profile['time_window']
            st.write(f"**Time:** {tw.get('start', 'N/A')} - {tw.get('end', 'N/A')}")
        if profile.get('party'):
            party = profile['party']
            st.write(f"**Party Size:** {party.get('adults_count', 0)} adults, {party.get('children_count', 0)} children")
    
    with col2:
        if profile.get('language'):
            st.write(f"**Language:** {profile['language']}")
        if profile.get('interests'):
            interests = [k for k, v in profile['interests'].items() if v > 0]
            if interests:
                st.write(f"**Interests:** {', '.join(interests[:3])}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_itinerary(itinerary):
    """Display the complete itinerary"""
    st.markdown('<div class="itinerary-section">', unsafe_allow_html=True)
    st.markdown("## 🗓️ Your Personalized Itinerary")
    
    # Summary
    summary = itinerary.get('summary', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Duration", f"{summary.get('total_duration_minutes', 0)} min")
    with col2:
        st.metric("Places to Visit", summary.get('total_places', 0))
    with col3:
        st.metric("Objects of Interest", summary.get('total_objects', 0))
    
    # Weather info
    if summary.get('weather_consideration'):
        st.info(f"☀️ {summary['weather_consideration']}")
    
    st.markdown("---")
    
    # Time slots
    for i, slot in enumerate(itinerary.get('itinerary', []), 1):
        st.markdown(f'<div class="time-slot">', unsafe_allow_html=True)
        
        st.markdown(f"### ⏰ {slot['time_slot']}")
        st.markdown(f"**📍 {slot['group']['title']}** - {slot['subgroup']['title']}")
        
        # Places in this slot
        for place_info in slot['places']:
            place = place_info['place']
            duration = place_info['suggested_duration_minutes']
            
            with st.expander(f"🏛️ {place['title']} ({duration} min)"):
                st.write(f"**Priority Score:** {place['priority']:.2f}")
                st.write(f"**Objects to see:** {place['object_count']}")
                
                # Highlights
                if place_info['highlights']:
                    st.markdown("**Top Highlights:**")
                    for obj in place_info['highlights']:
                        st.markdown(f"- {obj['title']} (score: {obj['priority']:.2f})")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# --- Logique Principale de l'Application ---

if st.session_state.phase == "greeting":
    st.markdown("### Bienvenue au Guide Royal ! Décrivez-moi votre visite idéale.")
    user_input = st.text_area("Votre description :", height=150, key="initial_input")

    if st.button("Planifier ma visite", type="primary"):
        if user_input.strip():
            st.session_state.messages_for_display.append({"role": "user", "content": user_input})
            with st.spinner("Analyse de votre demande..."):
                # On passe le compteur de tentatives (0 au début)
                result = invoke_graph(HumanMessage(content=user_input), st.session_state.question_attempts)
                if result:
                    st.session_state.current_graph_state = result
                    st.session_state.question_attempts = result.get('question_attempts', 0)
                    missing = result.get('user_profile', {}).get('missing_fields')

                    if missing:
                        st.session_state.phase = "clarification"
                        ai_question = result['messages'][-1].content
                        st.session_state.messages_for_display.append({"role": "assistant", "content": ai_question})
                    else:
                        st.session_state.phase = "processing"
                    st.rerun()

elif st.session_state.phase == "clarification":
    display_chat_history()

    # Utiliser st.chat_input pour une interface de chat standard
    if user_response := st.chat_input("Votre réponse..."):
        st.session_state.messages_for_display.append({"role": "user", "content": user_response})

        with st.spinner("Traitement..."):
            # On passe le compteur de tentatives actuel
            result = invoke_graph(HumanMessage(content=user_response), st.session_state.question_attempts)
            if result:
                st.session_state.current_graph_state = result
                # CORRECTION: Mettre à jour le compteur avec la nouvelle valeur retournée par le graphe
                st.session_state.question_attempts = result.get('question_attempts', 0)
                missing = result.get('user_profile', {}).get('missing_fields')

                # CORRECTION: La logique de décision utilise maintenant le compteur géré par Streamlit
                if missing and st.session_state.question_attempts < st.session_state.MAX_ATTEMPTS:
                    ai_question = result['messages'][-1].content
                    st.session_state.messages_for_display.append({"role": "assistant", "content": ai_question})
                else:
                    if not missing:
                        st.success("Profil complet ! Préparation de votre itinéraire...")
                    else:
                        st.warning("Nombre maximum de tentatives atteint. Je continue avec les informations disponibles.")
                    st.session_state.phase = "processing"
                
                st.rerun()

elif st.session_state.phase == "processing":
    st.markdown("## Création de votre itinéraire personnalisé...")
    
    # Préparer les éléments de l'interface pour les mises à jour en temps réel
    progress_bar = st.progress(0, text="Initialisation du traitement...")
    status_text = st.empty()
    
    # On doit fournir une entrée au graphe pour le démarrer.
    # Même si le profil est complet, le graphe commence par l'extracteur,
    # qui a besoin d'un message. On peut envoyer un message de contexte.
    # Le routeur s'assurera de nous envoyer directement vers les outils.
    context_message = HumanMessage(content="Le profil est maintenant complet. Veuillez procéder à la création de l'itinéraire.")

    # Préparer les arguments pour l'invocation
    inputs = {
        "messages": [context_message],
        "question_attempts": st.session_state.question_attempts
    }
    
    final_graph_state = None
    
    # Utiliser graph.stream() pour obtenir des mises à jour en temps réel
    try:
        status_text.text("Finalisation du profil et routage vers les outils...")
        progress_bar.progress(10)

        for event in graph.stream(inputs, st.session_state.config):
            # event est un dictionnaire de la forme {"nom_du_noeud": etat_apres_noeud}
            node_name = list(event.keys())[0]
            node_output = list(event.values())[0]
            
            # Mettre à jour l'interface en fonction du nœud qui a terminé
            if node_name == "rag_search_node":
                status_text.text("✅ Recherche des lieux d'intérêt terminée.")
                progress_bar.progress(40)
            elif node_name == "weather_node":
                status_text.text("✅ Données météorologiques récupérées.")
                progress_bar.progress(60)
            elif node_name == "planning_node":
                status_text.text("⏳ Construction de votre itinéraire personnalisé...")
                progress_bar.progress(80)
            elif node_name == "synthesis_node":
                status_text.text("✅ Itinéraire finalisé !")
                progress_bar.progress(100)
            
            # Conserver le dernier état complet retourné
            final_graph_state = node_output

        # Une fois le stream terminé, on a le résultat final
        if final_graph_state:
            st.session_state.current_graph_state = final_graph_state
            st.session_state.phase = "complete"
            st.success("Votre itinéraire est prêt !")
            import time
            time.sleep(1) # Petite pause pour que l'utilisateur voie le message de succès
            st.rerun()
        else:
            st.error("Le processus de planification n'a pas pu être complété.")

    except Exception as e:
        st.error(f"Une erreur critique est survenue durant la planification : {e}")


elif st.session_state.phase == "complete":
    result = st.session_state.current_graph_state
    
    if result.get('user_profile'):
        display_profile(result['user_profile'])
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📅 Itinéraire", "🏛️ Lieux Correspondants", "📊 Données Brutes"])
    
    collected = result.get('collected_data', {})
    
    with tab1:
        if 'itinerary' in collected and 'error' not in collected['itinerary']:
            display_itinerary(collected['itinerary'])
        else:
            st.warning("L'itinéraire n'a pas pu être généré. Essayez de fournir plus de détails.")
            if 'itinerary' in collected and 'error' in collected['itinerary']:
                st.error(f"Raison : {collected['itinerary']['error']}")

    # ... (les autres onglets restent inchangés)

    st.markdown("---")
    if st.button("🔄 Planifier une nouvelle visite"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Barre Latérale ---
with st.sidebar:
    st.markdown("### À propos du Guide Royal")
    st.markdown("Votre planificateur de visite personnel pour Versailles.")
    st.markdown("---")
    st.markdown("### Infos de Session")
    st.write(f"**Phase:** {st.session_state.phase}")
    # CORRECTION: Afficher le compteur de tentatives géré par Streamlit
    st.write(f"**Tentatives de clarification:** {st.session_state.question_attempts} / {st.session_state.MAX_ATTEMPTS}")

    # Affiche les champs manquants en temps réel
    if st.session_state.phase == "clarification":
        profile = st.session_state.current_graph_state.get('user_profile', {})
        missing = profile.get('missing_fields', [])
        if missing:
            st.markdown("**Infos manquantes :**")
            for field in missing:
                st.write(f"- `{field}`")