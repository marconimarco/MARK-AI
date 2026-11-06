import os
import re
import wave
import tempfile
import asyncio
import multiprocessing
import gradio as gr
import webbrowser
import requests
import numpy as np
from threading import Timer, Lock
from huggingface_hub import hf_hub_download

try:
    print("[MARK SYSTEM] Avvio interfaccia...")

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass

    RAG_DISPONIBILE = False
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        from pypdf import PdfReader
        RAG_DISPONIBILE = True
    except ImportError:
        pass

    AUDIO_DISPONIBILE = False
    try:
        from audio_transcriber import transcribe
        AUDIO_DISPONIBILE = True
    except ImportError:
        pass

    TTS_DISPONIBILE = False
    try:
        import edge_tts
        TTS_DISPONIBILE = True
    except ImportError:
        pass

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output").replace("\\", "/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================
    # CONFIGURAZIONE HUGGING FACE (Pubblica, senza token)
    # ========================================================
    REPO_ID = "DanDanDan9/MARK-AI"
    FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
    # ========================================================

    llm = None
    llm_lock = Lock()

    def carica_modello():
        global llm
        if llm is not None: return llm
        with llm_lock:
            if llm is not None: return llm
            
            try:
                print(f"[MARK] Controllo/Download del modello {FILENAME} da Hugging Face...")
                # Download anonimo (senza token)
                percorso_modello_scaricato = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            except Exception as e:
                print(f"[ERRORE] Download da Hugging Face fallito: {e}")
                return None

            try:
                from llama_cpp import Llama
                n_threads = max(1, multiprocessing.cpu_count() // 2)
                print("[MARK] Caricamento modello GGUF in memoria...")
                llm = Llama(
                    model_path=percorso_modello_scaricato,
                    n_ctx=1024,
                    n_threads=n_threads,
                    use_mmap=True,
                    use_mlock=True,
                    verbose=False
                )
                print("[MARK] Modello pronto.")
                return llm
            except Exception as e:
                print(f"[ERRORE] Modello: {e}")
                return None

    embedder = None
    indice_faiss = None
    chunks_documenti = []

    def carica_documento(file_path):
        global embedder, indice_faiss, chunks_documenti
        if not RAG_DISPONIBILE or not file_path: return ""
        try:
            if embedder is None: embedder = SentenceTransformer("all-MiniLM-L6-v2")
            testo = "\n".join((p.extract_text() or "") for p in PdfReader(file_path).pages)
            chunks = [testo[i:i + 500].strip() for i in range(0, len(testo), 450) if testo[i:i+500].strip()]
            if not chunks: return "Vuoto."
            emb = np.array(embedder.encode(chunks)).astype("float32")
            if indice_faiss is None: indice_faiss = faiss.IndexFlatL2(emb.shape[1])
            indice_faiss.add(emb)
            chunks_documenti.extend(chunks)
            return f"{os.path.basename(file_path)}"
        except Exception as e: 
            print(f"Errore RAG: {e}")
            return "Errore."

    def cerca_nel_documento(domanda, k=3):
        if not RAG_DISPONIBILE or indice_faiss is None: return ""
        try:
            q_emb = np.array(embedder.encode([domanda])).astype("float32")
            _, indici = indice_faiss.search(q_emb, min(k, len(chunks_documenti)))
            return "\n---\n".join(chunks_documenti[i] for i in indici[0] if i < len(chunks_documenti))
        except Exception as e:
            print(f"Errore ricerca RAG: {e}")
            return ""

    def svuota_documenti():
        global indice_faiss, chunks_documenti
        indice_faiss, chunks_documenti = None, []
        return ""

    PATTERN_IMG = re.compile(r"^\s*(?:per\s+favore\s+)?(disegna(?:mi)?|genera(?:mi)?|crea(?:mi)?|fai(?:mi)?|mostra(?:mi)?|abbozza(?:mi)?|immagina)\s+.+", re.IGNORECASE | re.DOTALL)
    PAROLE_ESCLUDI = re.compile(r"\b(cos'?è|che\s+cos|spiegami|parlami|dimmi|significato|differenza|come\s+funziona)\b", re.IGNORECASE)

    def richiede_immagine(t):
        t = t.strip()
        return bool(t and not PAROLE_ESCLUDI.search(t) and (PATTERN_IMG.match(t) or "immagine di" in t.lower() or "foto di" in t.lower()))

    def genera_immagine(prompt):
        try:
            prompt = re.sub(r"^\s*(?:per\s+favore\s+)?(disegna(?:mi)?|genera(?:mi)?|crea(?:mi)?|fai(?:mi)?|mostra(?:mi)?|abbozza(?:mi)?|immagina)\s+", "", prompt, flags=re.IGNORECASE)
            prompt = re.sub(r"^(un[a]?\s+)?(immagine|foto|disegno|illustrazione)\s+(di|con)\s+", "", prompt, flags=re.IGNORECASE).strip()
            url = f"https://image.pollinations.ai/p/{requests.utils.quote(prompt)}?width=768&height=768&seed=42&nologo=true&model=flux"
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                path = os.path.join(OUTPUT_DIR, f"img_{abs(hash(prompt))%10000000}.png")
                with open(path, "wb") as f: f.write(r.content)
                return path
        except: pass
        return None

    def trascrivi_audio(audio_data):
        if not AUDIO_DISPONIBILE or audio_data is None: return ""
        path = None
        try:
            if isinstance(audio_data, tuple):
                sr, y = audio_data
                y = np.asarray(y, dtype=np.float32).flatten()
                if y.size == 0: return ""
                y = (y / np.max(np.abs(y)) * 32767).astype(np.int16) if np.max(np.abs(y)) > 1.0 else (y * 32767).astype(np.int16)
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sr)); wf.writeframes(y.tobytes())
            else: path = str(audio_data)
            res = transcribe(path, model="small", language="it")
            return (getattr(res, "text", str(res))).strip()
        except: return ""
        finally:
            if path and os.path.exists(path) and path.startswith(tempfile.gettempdir()): os.remove(path)

    VOCI = {"Isabella (profonda)": "it-IT-IsabellaNeural", "Elsa (neutra)": "it-IT-ElsaNeural"}

    def sintetizza_voce(testo, voce_label, velocita):
        if not TTS_DISPONIBILE or not testo.strip(): return None
        try:
            file_out = os.path.join(OUTPUT_DIR, f"tts_{abs(hash(testo))%10000000}.mp3")
            asyncio.run(edge_tts.Communicate(testo, VOCI.get(voce_label, "it-IT-IsabellaNeural"), rate=f"{velocita:+d}%", pitch="-5Hz").save(file_out))
            return file_out
        except: return None

    def assistente_unificato(msg, chat):
        chat = list(chat) if chat else []
        if not msg.strip(): return chat, ""
        modello = carica_modello()
        
        if any(k in msg.lower() for k in ["documento", "pdf"]) and RAG_DISPONIBILE:
            ctx = cerca_nel_documento(msg)
            prompt = f"<|im_start|>system\nRispondi in italiano usando SOLO il contesto fornito.<|im_end|>\n<|im_start|>user\nContesto:\n{ctx}\n\nDomanda: {msg}<|im_end|>\n<|im_start|>assistant\n"
        elif richiede_immagine(msg):
            chat.append(gr.ChatMessage(role="user", content=msg))
            chat.append(gr.ChatMessage(role="assistant", content=""))
            yield chat, ""
            img = genera_immagine(msg)
            chat[-1] = gr.ChatMessage(role="assistant", content={"path": img, "alt_text": msg}) if img else gr.ChatMessage(role="assistant", content="Errore.")
            yield chat, ""
            return
        else:
            prompt = f"<|im_start|>system\nSei MARK.<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
            
        if modello is None:
            chat.extend([gr.ChatMessage(role="user", content=msg), gr.ChatMessage(role="assistant", content="Modello non disponibile (download fallito o in corso).")])
            yield chat, ""
            return
            
        chat.extend([gr.ChatMessage(role="user", content=msg), gr.ChatMessage(role="assistant", content="")])
        yield chat, ""
        
        txt = ""
        try:
            for chunk in modello(prompt, max_tokens=400, temperature=0.6, stop=["<|im_end|>"], stream=True):
                snip = chunk.get("choices", [{}])[0].get("text", "") if isinstance(chunk, dict) else getattr(chunk.choices[0], "text", "")
                if snip:
                    txt += snip
                    chat[-1] = gr.ChatMessage(role="assistant", content=txt)
                    yield chat, ""
        except: pass

    def audio_a_chat(audio_data, chat):
        chat = list(chat) if chat else []
        testo = trascrivi_audio(audio_data)
        if not testo: yield chat, ""; return
        for c, _ in assistente_unificato(testo, chat): yield c, ""

    CSS_CUSTOM = """
    body, .gradio-container {
        background-color: #000000 !important;
        color: #00f3ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 13px !important;
    }
    textarea, input[type="text"] {
        background-color: #0a0a0a !important;
        border: 1px solid #00f3ff !important;
        color: #ffffff !important;
        font-size: 13px !important;
    }
    button {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        color: #cccccc !important;
        font-size: 12px !important;
    }
    button:hover {
        border-color: #00f3ff !important;
        color: #00f3ff !important;
    }
    .chatbot, .chatbot * {
        background-color: #050505 !important;
        border: 1px solid #222222 !important;
        font-size: 13px !important;
    }
    .gr-file-dropzone {
        min-height: 30px !important;
        max-height: 30px !important;
        padding: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .gr-file-dropzone *, .gr-file-dropzone span, .gr-file-dropzone div, .gr-file-dropzone p {
        font-size: 8px !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: center !important;
    }
    .box-documenti {
        max-width: 100% !important;
        max-height: 33% !important;
        overflow: hidden !important;
    }
    .box-audio {
        max-width: 100% !important;
    }
    """

    with gr.Blocks(title="M.A.R.K. AI") as demo:
        gr.HTML(f"<style>{CSS_CUSTOM}</style>")
        
        gr.Markdown("# M.A.R.K. AI")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="M.A.R.K. Core Chat", height=500)
                input_testo = gr.Textbox(label="Comando:", lines=1)
                with gr.Row():
                    bottone_invia = gr.Button("Invia", variant="primary", size="sm")
                    bottone_stop = gr.Button("Interrompi", variant="stop", size="sm")

            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="box-documenti"):
                        gr.Markdown("📄 Doc")
                        file_pdf = gr.File(label="", file_types=[".pdf"], type="filepath", show_label=False, height=30)
                        stato_doc = gr.Textbox(label="", show_label=False, interactive=False, lines=1)
                        btn_svuota = gr.Button("X", size="sm")
                        file_pdf.change(fn=carica_documento, inputs=file_pdf, outputs=stato_doc)
                        btn_svuota.click(fn=svuota_documenti, inputs=None, outputs=stato_doc)

                    with gr.Column(scale=1, elem_classes="box-audio"):
                        gr.Markdown("🎤 Audio")
                        audio_input = gr.Audio(sources="microphone", type="numpy", show_label=False)
                        stato_audio = gr.Textbox(label="", show_label=False, interactive=False, lines=1)
                        audio_input.stop_recording(fn=audio_a_chat, inputs=[audio_input, chatbot], outputs=[chatbot, stato_audio], show_progress="hidden")

                with gr.Accordion("🔊 Voce", open=False):
                    testo_tts = gr.Textbox(label="", show_label=False, lines=2)
                    voce_tts = gr.Dropdown(choices=list(VOCI.keys()), value="Isabella (profonda)", show_label=False)
                    velocita_tts = gr.Slider(minimum=-50, maximum=50, value=0, step=10, show_label=False)
                    bottone_tts = gr.Button("Genera", size="sm")
                    audio_output = gr.Audio(label="", show_label=False, type="filepath", interactive=False)
                    stato_tts = gr.Textbox(label="", show_label=False, interactive=False)
                    bottone_tts.click(fn=sintetizza_voce, inputs=[testo_tts, voce_tts, velocita_tts], outputs=[audio_output, stato_tts])

        evento_click = bottone_invia.click(
            fn=assistente_unificato, 
            inputs=[input_testo, chatbot], 
            outputs=[chatbot, input_testo]
        )
        evento_submit = input_testo.submit(
            fn=assistente_unificato, 
            inputs=[input_testo, chatbot], 
            outputs=[chatbot, input_testo]
        )
        bottone_stop.click(fn=lambda: (gr.update(), gr.update()), inputs=None, outputs=[chatbot, input_testo], cancels=[evento_click, evento_submit], queue=False)

    def apri_browser():
        webbrowser.open("http://127.0.0.1:7860")

    if __name__ == "__main__":
        Timer(2, apri_browser).start()
        demo.queue().launch(share=False, server_port=7860, inbrowser=False, allowed_paths=[OUTPUT_DIR])

except Exception as e:
    import traceback
    traceback.print_exc()
    print("\n\n=== ERRORE CRITICO ALL'AVVIO ===")
    print(f"Tipo: {type(e).__name__}")
    print(f"Messaggio: {e}")
    print("================================")
    input("\nPremi INVIO per chiudere...")
