import os
import requests
import json
import time
from project.config import OPENAI_API_KEY, GOOGLE_API_KEY, LLM_PROVIDER, MODEL_NAME, LLM_REQUEST_DELAY, LLM_MAX_RETRIES, LLM_RETRY_BASE_WAIT

def call_llm(prompt, system_role="You are a helpful assistant."):
    """
    Unified function to call GPT-4o-mini logic.
    Includes Rate Limiting and Automatic Retries.
    """
    # --- Configuration Check ---
    if LLM_PROVIDER == "google":
        key = GOOGLE_API_KEY
        if not key or "YOUR" in key:
            print("[LLM Utils] No Google API Key. Mocking.")
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={key}"
        # Gemini Payload
        # Note: system_instruction is available in beta for some models, or put in first user part.
        # "system_instruction": {"parts": {"text": system_role}}
        payload = {
            "contents": [{
                "parts": [{"text": f"{system_role}\n\n{prompt}"}] 
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 8192 # Increased for large batches. Note: Gemini Flash output limit might be 8k.
            }
        }
        headers = {"Content-Type": "application/json"}
    else:
        # OpenAI
        key = OPENAI_API_KEY
        if not key or "YOUR" in key:
            print("[LLM Utils] No OpenAI API Key. Mocking.")
            return None
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 16383 # Max for GPT-4o-mini
        }

    for attempt in range(LLM_MAX_RETRIES):
        try:
            # Rate Limit Delay (Throttle)
            time.sleep(LLM_REQUEST_DELAY) 
            
            response = requests.post(url, headers=headers, json=payload)
            
            # Handle 429 specifically
            if response.status_code == 429:
                wait_time = LLM_RETRY_BASE_WAIT * (2 ** attempt)
                print(f"[LLM Utils] Rate Limit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            res_json = response.json()
            
            # Parse Response based on Provider
            if LLM_PROVIDER == "google":
                # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
                try:
                    content = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                except (KeyError, IndexError):
                    print(f"[LLM Utils] Gemini Parse Error: {res_json}")
                    return None
            else:
                # OpenAI
                content = res_json['choices'][0]['message']['content'].strip()
                
            return content
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401 or e.response.status_code == 403:
                print(f"[LLM Utils] Auth Error ({e.response.status_code}). Check Key.")
                return None 
            print(f"[LLM Utils] HTTP Error: {e}. Retrying...")
            
        except Exception as e:
            print(f"[LLM Utils] Error: {e}")
            
    print("[LLM Utils] Max retries exceeded.")
    return None
