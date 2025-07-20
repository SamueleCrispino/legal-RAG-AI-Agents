import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import vertexai

env_path: str = "../.env"
load_dotenv(env_path)

API_KEY_VAR = "GEMINI_2_5_PRO_KEY"


def load_api_key(var_name: str = "GEMINI_2_5_PRO_KEY") -> str:
    """
    Prova a caricare la variabile API da un file .env. 
    Usa python-dotenv se installato, altrimenti fa parse manuale.
    """
    # Metodo con python-dotenv, se disponibile:
    try:
        key = os.getenv(var_name)
        if key:
            return key
    except ImportError:
        pass

    raise KeyError(f"Variabile {var_name} non trovata in {env_path}")


genai.configure(api_key=load_api_key(var_name=API_KEY_VAR))


def generate_gemini_content(
    prompt_text: str,
    model: str = "gemini-2.0-flash",
    env_path: str = "../.env",
    api_key_var: str = "GEMINI_2_5_PRO_KEY",
    max_retries: int = 3,
    timeout: float = 30.0
):
    """
    Invoca l’endpoint generateContent di Gemini usando API Key caricata da .env.
    Restituisce la stringa di risposta come dict.
    """
    api_key = load_api_key(var_name=api_key_var)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            if attempt < max_retries:
                continue
            else:
                raise RuntimeError(f"Errore di rete dopo {attempt} tentativi: {e}")
        if resp.status_code == 200:
            try:
                resp_json = resp.json()
                
                return resp_json["candidates"][0]["content"]["parts"][0]["text"]
            except json.JSONDecodeError:
                raise ValueError("Risposta non in formato JSON valido")
        else:
            # Se status code non 200, log e riprova se possibile
            msg = f"Status {resp.status_code}: {resp.text}"
            if attempt < max_retries:
                continue
            else:
                raise RuntimeError(f"Chiamata API fallita dopo {attempt} tentativi: {msg}")
    # Non dovrebbe arrivare qui
    raise RuntimeError("Fallito a invocare Gemini API")