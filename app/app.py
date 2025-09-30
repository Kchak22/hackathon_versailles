# app.py (corrigé)
import os
import io
import hashlib
import datetime
from pathlib import Path
import html


import streamlit as st
from pydub import AudioSegment        # nécessite ffmpeg
from voice_agent import get_text_from_voice  # ta fonction côté agent

import imageio_ffmpeg as iio_ffmpeg


# indique à pydub quel exécutable ffmpeg utiliser (fourni par imageio-ffmpeg)
AudioSegment.converter = iio_ffmpeg.get_ffmpeg_exe()

st.set_page_config(page_title="Assistant d'enregistrement — Parcours Versailles", layout="centered")

OUT_DIR = Path("recordings")
OUT_DIR.mkdir(exist_ok=True)

# ---------- Styles (cartouche noir) ----------
st.markdown(
    """
    <style>
    .top-card {
        background: linear-gradient(180deg,black,#1a1a1a);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        margin-bottom: 20px;
        color: black;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .top-card h3 { margin: 0 0 6px 0; font-size: 20px; color: #fff; }
    .top-card p { margin: 0; color: #ddd; opacity: 0.95; }

    div.stButton > button:first-child {
        border-radius: 999px;
        height: 92px;
        width: 92px;
        font-size: 38px;
        border: none;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        background: linear-gradient(180deg,#444444,#222222);
        color: white;
        transition: transform 0.08s ease-in-out;
    }
    div.stButton > button:first-child:active { transform: scale(0.98); }

    .helper { text-align:center; color: black; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-card">
        <h3>Assistant d'enregistrement — Parcours Versailles</h3>
        <p>Enregistrez un message vocal — l'audio sera sauvegardé puis envoyé à l'agent pour transcription (automatique à la fin de l'enregistrement).</p>
    </div>
    """,
    unsafe_allow_html=True
)

# session state init
st.session_state.setdefault("show_recorder", True)
st.session_state.setdefault("first_done", False)       # False until first recording processed
st.session_state.setdefault("second_done", False)      # False until second recording processed
st.session_state.setdefault("response_1", None)        # texte renvoyé par get_text_from_voice (1ère itération)
st.session_state.setdefault("response_2", None)        # texte renvoyé par get_text_from_voice (2ième itération)
st.session_state.setdefault("last_audio_hash_1", None)
st.session_state.setdefault("last_audio_hash_2", None)
st.session_state.setdefault("chat_history", [])      

# mic button
col1, col2, col3 = st.columns([1, 0.4, 1])

st.session_state.setdefault("show_recorder", True)
st.session_state.setdefault("first_done", False)
st.session_state.setdefault("second_done", False)
st.session_state.setdefault("response_1", None)
st.session_state.setdefault("response_2", None)
st.session_state.setdefault("last_audio_hash_1", None)
st.session_state.setdefault("last_audio_hash_2", None)

# affiche les réponses déjà reçues (si existantes)
def render_bot_responses():
    for i, resp in enumerate([st.session_state.get("response_1"), st.session_state.get("response_2")], start=1):
        if resp:
            safe = html.escape(resp).replace("\n", "<br>")
            try:
                with st.chat_message("assistant"):
                    st.markdown(
                        f'<div style="background:#e6e6e6;color:#111;padding:12px;border-radius:12px;max-width:80%;">'
                        f'{safe}</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.markdown(f"**Assistant (réponse {i}) :** {resp}")

render_bot_responses()
st.markdown("---")

# Si deux appels déjà faits, on désactive l'enregistreur
if st.session_state["first_done"] and st.session_state["second_done"]:
    st.success("Les deux enregistrements ont été traités. Plus d'enregistrement possible.")
else:
    # choisir la clé audio selon l'itération en cours
    if not st.session_state["first_done"]:
        audio_key = "audio_input_1"
        call_idx = 1
    else:
        audio_key = "audio_input_2"
        call_idx = 2

    # widget natif
    audio_obj = None
    if st.session_state.get("show_recorder"):
        st.write("### Enregistrement")
        st.write("Appuie sur **Record** puis **Stop**. À la fin, la réponse du bot s'ajoute dessous.")
        try:
            audio_obj = st.audio_input("Appuie sur Record puis Stop.", key=audio_key)
        except Exception:
            st.error("Votre version de Streamlit n'a pas `st.audio_input`. Je peux fournir une alternative HTML/JS si nécessaire.")
            audio_obj = None

    # traitement
    if audio_obj:
        try:
            if hasattr(audio_obj, "read"):
                raw_bytes = audio_obj.read()
            elif isinstance(audio_obj, (bytes, bytearray)):
                raw_bytes = bytes(audio_obj)
            else:
                raw_bytes = audio_obj.getvalue()
        except Exception as e:
            st.error(f"Impossible de lire l'enregistrement : {e}")
            raw_bytes = None

        if raw_bytes:
            # hash & prevention double-processing
            audio_hash = hashlib.sha256(raw_bytes).hexdigest()
            last_hash_key = "last_audio_hash_1" if call_idx == 1 else "last_audio_hash_2"
            response_key = "response_1" if call_idx == 1 else "response_2"

            if st.session_state.get(last_hash_key) == audio_hash:
                st.info("Cet enregistrement a déjà été traité.")
            else:
                st.session_state[last_hash_key] = audio_hash

                # Appel à get_text_from_voice avec call_idx
                with st.spinner("Je réfléchis..."):
                    try:
                        API_KEY = "hNI723eridx5FEd22retuvetk6wjXxsv"
                        AGENT_ID = "ag:a30ad7d2:20250929:voice-to-text:94223d91"
                        # IMPORTANT: get_text_from_voice doit accepter call_idx (voir B)
                        result = get_text_from_voice(raw_bytes, api_key=API_KEY, agent_id=AGENT_ID, call_idx=call_idx)
                    except Exception as e:
                        result = {"success": False, "error": f"Exception appel agent: {e}"}

                ts_now = datetime.datetime.now().strftime("%H:%M:%S")
                if not result.get("success"):
                    reply_text = f"⚠️ {result.get('error') or 'Erreur pendant la transcription.'}"
                else:
                    reply_text = result.get("text") or "(Aucune transcription automatique)"

                # stocker la réponse correspondant à l'itération
                st.session_state[response_key] = reply_text

                # afficher immédiatement la réponse (ajoute sous les précédentes)
                safe = html.escape(reply_text).replace("\n", "<br>")
                try:
                    with st.chat_message("assistant"):
                        st.markdown(
                            f'<div style="background:#e6e6e6;color:#111;padding:12px;border-radius:12px;max-width:80%;">'
                            f'{safe}<div style="font-size:10px;color:#666;margin-top:6px">{ts_now}</div></div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    st.markdown(f"**Assistant :** {reply_text}")

                # cleanup: supprimer la valeur du widget pour forcer un nouvel enregistrement propre
                try:
                    del st.session_state[audio_key]
                except Exception:
                    pass

                # marquer iteration comme faite
                if call_idx == 1:
                    st.session_state["first_done"] = True
                else:
                    st.session_state["second_done"] = True

                # si on a fini les deux, désactiver le recorder
                if st.session_state["first_done"] and st.session_state["second_done"]:
                    st.session_state["show_recorder"] = False

                # enfin on force un rerun propre pour que l'UI se réaffiche (fallback st.stop si nécessaire)
                try:
                    st.experimental_rerun()
                except Exception:
                    st.stop()