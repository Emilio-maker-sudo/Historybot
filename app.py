import os
import re
import time
import logging
import gradio as gr
import numpy as np
import requests
from moviepy.editor import *
from pydub import silence
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from elevenlabs import generate, save, set_api_key

# Configuración inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar API Keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
set_api_key(ELEVENLABS_API_KEY)

# Directorios para recursos
SFX_DIR = "sfx"
MUSIC_DIR = "music"
FONT_PATH = "font.ttf"

# Mapeo de efectos de sonido (SFX_MAPPING)
SFX_MAPPING = {
    # Efectos generales
    "afilado": "AFILADO.mp3",
    "alerta": "Alerta.mp3",
    "ternura": "Awww (ternura).mp3",
    "bromita": "Ba dum ts.mp3",
    "disparo": "Disparo.mp3",
    "headshot": "Bum headshoot.mp3",
    "risa": "Risa Troll.mp3",
    "risa malvada": "Risa Malvada.mp3",
    "aplausos": "Publico gritando y aplaudiendo.mp3",
    "suspenso": "Suspenso Drama.mp3",
    "revelación": "Suspenso para Revelar algo.mp3",
    "transición": "SWOOSH.mp3",
    "transición épica": "SWOOSH - DRAMATIC.mp3",
    "golpe": "Golpe.mp3",
    "cachetada": "Cachetada.mp3",
    "censura": "Censura.mp3",
    "fatality": "Fatality.mp3",
    "misión cumplida": "Mision Passed San Andreas.mp3",
    "momento épico": "Momento épico.mp3",
    "sorpresa": "SORPRESA.mp3",
    "terror": "Sonido de terror.mp3",
    "incendio": "Incendio.mp3",
    "latido": "Latido de Corazon.mp3",
    "licuadora": "Licuadora.mp3",
    "minigun": "Minigun.mp3",
    "moneda": "Moneda de Mario.mp3",
    "reloj": "Reloj.mp3",
    "teletransportación": "Teletransportación Goku.mp3",
    "timbre": "Timbre de casa.mp3",
    "foto": "Tomar foto.mp3",
    "cristal": "cristal rompiendose.mp3",
    "ronquido": "persona roncando.mp3",
    "patito": "sonido de patito.mp3",
    "tun tun": "tun tun tunnnnnnn.mp3",
    "turi ip ip": "turi ip ip ip.mp3",
    "trompeta": "chan chan chan.mp3",
    "sword": "SWORD CLANG.mp3",
    "espada": "SWORD DRAW.mp3",
    "risa pública": "risas del publico.mp3",
    "música épica": "música de orquesta.mp3",
    "tragedia": "PREtragedia.mp3",
    "interrupción": "Interrupcion.mp3",
    "estático": "TV STATIC.mp3",
    "tambores": "Tambores.mp3",
    "flauta triste": "Flauta Sad.mp3",
    "grillo": "Grillo.mp3",
    "estómago": "Estomago vacío.mp3",
    "noticiero": "Entrada noticiera Latina PERÚ.mp3",
    "disco rayado": "Efecto de Disco Rayado.mp3",
    "kirby": "Conchetumare kirby.mp3",
    "cuack": "Cuack.mp3",
    "deez nutz": "Deez nutz.mp3",
    "gay": "HA! GAAAAY.mp3",
    "grito": "Grito AHHHAHH.mp3",
    "persecución": "Grito persecusión.mp3",
    "swish": "Swish swoosh.mp3",
    "pop": "Sonido Pop Para texto.mp3",
    "texto": "Text Pop Up - Sound Effect [HD](MP3_128K).mp3",
    "transición 2": "Transition - Sound Effect [HD](MP3_128K).mp3",
    "ta dah": "TA DAHH.mp3",
    "tiempo parado": "TIEMPO PARADO.mp3",
    "tutorial": "Tutorial2.0.mp3",
    "llamada": "llamada sansumg.mp3",
    "escuela": "timbre de escuela.mp3",
    "cinta": "cinta rebobinando.mp3",
}

def generar_guion(descripcion):
    """Genera un guion llamativo usando GPT-4"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
    Crea un guion para un short de YouTube de 30-60 segundos basado en esta descripción:
    {descripcion}

    El guion debe:
    - Empezar con un gancho impactante
    - Ser enganchador, dinámico y con elementos de suspenso
    - Incluir indicaciones temporales para sincronizar con video
    - Dividirse en segmentos de 5-10 segundos
    - Incluir 3-5 efectos de sonido relevantes
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    return response.choices[0].message.content

def procesar_video(video_path, guion):
    """Procesa el video según el guion generado"""
    # 1. Detectar silencios y partes útiles
    video = VideoFileClip(video_path)
    audio = video.audio.to_soundarray()
    
    # Detectar segmentos no silenciosos
    non_silent = silence.detect_nonsilent(
        audio, 
        min_silence_len=500,
        silence_thresh=-40
    )
    
    # 2. Recortar video
    clips = []
    for start, end in non_silent:
        clips.append(video.subclip(start/1000, end/1000))
    
    edited_video = concatenate_videoclips(clips)
    
    # 3. Dividir guion en segmentos
    segments = []
    for line in guion.split("\n"):
        if "[" in line and "]" in line:
            time_match = re.search(r"\[(\d+-\d+)\]", line)
            text = re.sub(r"\[.*?\]", "", line).strip()
            segments.append({
                "text": text,
                "duration": time_match.group(1) if time_match else "5-8"
            })
    
    # 4. Generar elementos visuales y de audio
    final_clips = []
    for i, segment in enumerate(segments):
        # Texto animado
        txt_clip = TextClip(
            segment["text"],
            fontsize=60,
            color="white",
            font=FONT_PATH,
            stroke_color="black",
            stroke_width=2
        ).set_position(("center", "top"))
        
        # Duración del segmento
        min_d, max_d = map(int, segment["duration"].split("-"))
        duration = random.uniform(min_d, max_d)
        
        # Efectos de sonido
        sfx = None
        for keyword in SFX_MAPPING:
            if keyword in segment["text"].lower():
                sfx = AudioFileClip(os.path.join(SFX_DIR, SFX_MAPPING[keyword]))
                break
        
        # Combinar elementos
        video_clip = edited_video.subclip(i*5, (i+1)*5)
        composite = CompositeVideoClip([video_clip, txt_clip.set_duration(duration)])
        
        if sfx:
            composite = composite.set_audio(sfx)
        
        final_clips.append(composite)
    
    # 5. Agregar música de fondo
    music = AudioFileClip(random.choice(os.listdir(MUSIC_DIR))).volumex(0.3)
    final_video = concatenate_videoclips(final_clips)
    final_video = final_video.set_audio(music.set_duration(final_video.duration))
    
    return final_video

def generar_voz(texto):
    """Genera narración con ElevenLabs"""
    audio = generate(
        text=texto,
        voice="Bella",
        model="eleven_multilingual_v2"
    )
    
    filename = f"narracion_{int(time.time())}.mp3"
    save(audio, filename)
    return filename

def pipeline(input_video, descripcion):
    try:
        # 1. Generar guion
        guion = generar_guion(descripcion)
        logger.info(f"Guion generado:\n{guion}")
        
        # 2. Generar narración
        voz = generar_voz(guion)
        
        # 3. Procesar video
        video_final = procesar_video(input_video, guion)
        
        # 4. Exportar resultado
        output_path = f"short_{int(time.time())}.mp4"
        video_final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24
        )
        
        return output_path, guion
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None, f"Error: {str(e)}"

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🎬 Generador de Shorts Automáticos")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Sube tu video")
            descripcion = gr.Textbox(label="Descripción del contenido")
            generate_btn = gr.Button("Generar Short")
        
        with gr.Column():
            video_output = gr.Video(label="Short Final")
            guion_output = gr.Textbox(label="Guion Generado")
    
    generate_btn.click(
        fn=pipeline,
        inputs=[video_input, descripcion],
        outputs=[video_output, guion_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
