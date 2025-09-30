# voice_agent.py (version robuste, logs, diagnostique)
import os
import io
import base64
import datetime
import traceback
from pathlib import Path
import httpx
from pydub import AudioSegment

# tenter imageio-ffmpeg si présent
try:
    import imageio_ffmpeg as iio_ffmpeg
    ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg_exe and os.path.exists(ffmpeg_exe):
        AudioSegment.converter = ffmpeg_exe
        os.environ["FFMPEG_BINARY"] = ffmpeg_exe
except Exception:
    ffmpeg_exe = None

OUT_DIR = Path("recordings")
OUT_DIR.mkdir(exist_ok=True)


def to_wav_16k_mono_bytes(input_bytes: bytes) -> bytes:
    try:
        audio = AudioSegment.from_file(io.BytesIO(input_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)
        out = io.BytesIO()
        audio.export(out, format="wav")
        return out.getvalue()
    except FileNotFoundError as e:
        # erreur typique si ffmpeg/ffprobe manquant
        raise RuntimeError(f"ffmpeg/ffprobe introuvable ou non exécutable. Détails: {e}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la conversion audio: {e}")


def _content_to_text(content):
    if content is None:
        return None
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, dict):
        for k in ("text", "content", "value"):
            if k in content and isinstance(content[k], str):
                return content[k].strip() or None
        return None
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                for k in ("text", "content", "value"):
                    if k in item and isinstance(item[k], str):
                        parts.append(item[k].strip())
                        break
        return "\n".join([p for p in parts if p]) or None
    return None


def _extract_text_from_mistral_response(resp):
    # tente plusieurs chemins et renvoie (text_or_none, debug_dict)
    debug = {}
    try:
        # try choices style (chat.complete)
        if hasattr(resp, "choices"):
            debug["choices_attr"] = True
            try:
                choices = getattr(resp, "choices")
                if isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    message = getattr(first, "message", None)
                    if message is not None:
                        content = getattr(message, "content", None)
                        debug["choices_message_content"] = content
                        t = _content_to_text(content)
                        if t:
                            return t, debug
            except Exception as e:
                debug["choices_parse_error"] = repr(e)

        # try outputs style
        if hasattr(resp, "outputs"):
            debug["outputs_attr"] = True
            try:
                outs = getattr(resp, "outputs")
                debug["outputs_value"] = type(outs).__name__
                if isinstance(outs, (list, tuple)) and outs:
                    first = outs[0]
                    if isinstance(first, dict):
                        t = first.get("content") or first.get("text") or first.get("value")
                        if t:
                            return t, debug
                    elif isinstance(first, str):
                        return first, debug
                    else:
                        t = getattr(first, "content", None) or getattr(first, "text", None)
                        if t:
                            return t, debug
            except Exception as e:
                debug["outputs_parse_error"] = repr(e)

        # try dict-like
        if isinstance(resp, dict):
            debug["is_dict"] = True
            try:
                out = resp.get("output") or resp.get("outputs") or resp.get("choices")
                debug["dict_out"] = type(out).__name__ if out is not None else None
                if isinstance(out, list) and out:
                    itm = out[0]
                    if isinstance(itm, dict):
                        t = itm.get("content") or itm.get("text")
                        if t:
                            return t, debug
                    elif isinstance(itm, str):
                        return itm, debug
                t = resp.get("text") or resp.get("transcript") or resp.get("response")
                if t:
                    return t, debug
            except Exception as e:
                debug["dict_parse_error"] = repr(e)

        # try to_dict fallback
        if hasattr(resp, "to_dict"):
            try:
                d = resp.to_dict()
                debug["to_dict"] = True
                out = d.get("choices") or d.get("outputs") or d.get("output")
                if isinstance(out, list) and out:
                    c0 = out[0]
                    if isinstance(c0, dict):
                        msg = c0.get("message") or c0
                        cont = msg.get("content") if isinstance(msg, dict) else None
                        t = _content_to_text(cont)
                        if t:
                            return t, debug
                t = d.get("text") or d.get("transcript")
                if t:
                    return t, debug
            except Exception as e:
                debug["to_dict_error"] = repr(e)

    except Exception as e:
        debug["extract_exception"] = repr(e)

    return None, debug


def get_text_from_voice(v, api_key: str = None, agent_id: str = None, client: object = None, timeout: int = 60,call_idx: int = None):
    print(f"######################## called get_text_from_voice with call id : {call_idx    }")
    # resolver api/agent
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    agent_id = agent_id or os.environ.get("AGENT_ID")
    if not api_key:
        return {"success": False, "error": "MISTRAL_API_KEY manquant (passer api_key ou définir variable d'env)."}
    if not agent_id:
        return {"success": False, "error": "AGENT_ID manquant (passer agent_id ou définir variable d'env)."}

    # lire octets
    try:
        if hasattr(v, "read"):
            raw = v.read()
        elif isinstance(v, (bytes, bytearray)):
            raw = bytes(v)
        else:
            raw = v.getvalue()
    except Exception as e:
        return {"success": False, "error": f"Impossible de lire l'audio: {e}"}

    if not raw:
        return {"success": False, "error": "Aucun octet audio lu."}

    # conversion
    try:
        wav_bytes = to_wav_16k_mono_bytes(raw)
    except Exception as e:
        return {"success": False, "error": f"Erreur conversion audio (ffmpeg/pydub requis): {e}"}

    # save
    saved_path = None
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = OUT_DIR / f"recording_{ts}_16k_mono.wav"
        with open(saved_path, "wb") as f:
            f.write(wav_bytes)
    except Exception as e:
        saved_path = None

    # encode base64
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # init client
    try:
        if client is None:
            http_client = httpx.Client(timeout=timeout)
            from mistralai import Mistral
            client = Mistral(api_key=api_key, client=http_client)
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": f"Impossible d'initialiser client Mistral: {e}\n{tb}", "saved_path": str(saved_path) if saved_path else None}

    # prepare inputs (adapt if ton agent attend un schéma différent)
    inputs = [
        {"role": "system", "content": "Tu es un transcripteur. Réponds uniquement par le texte transcrit dans la langue que tu détectes."},
        {"role": "user", "content": {"type": "audio_base64", "data": audio_b64, "mime": "audio/wav", "instructions": "Transcris l'audio et renvoie uniquement le texte."}}
    ]

    # appel (essayer chat.complete d'abord si présent)
    try:
        # si la méthode chat.complete existe, l'utiliser (conforme doc voxtral sample)
        if hasattr(client, "chat") and hasattr(client.chat, "complete"):
            print("voice_agent: using client.chat.complete")
            chat_response = client.chat.complete(model="voxtral-mini-latest", messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": audio_b64}, {"type": "text", "text": "Transcris en francais et renvoie le texte brut."}]}])
            resp = chat_response
        else:
            print("voice_agent: using client.beta.conversations.start")
            resp = client.beta.conversations.start(agent_id=agent_id, inputs=inputs, store=False)
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": f"Erreur appel agent Mistral: {e}\n{tb}", "saved_path": str(saved_path) if saved_path else None}

    # extraction du texte
    text, debug = _extract_text_from_mistral_response(resp)
    print(f"############################### result")
    return {"success": True, "text": text, "raw": resp, "saved_path": str(saved_path) if saved_path else None, "extract_debug": debug}
