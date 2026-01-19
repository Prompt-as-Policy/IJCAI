import os
import pickle
import json
import time
from datetime import datetime
import sys

# Add project root to sys.path to fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests # Use requests to avoid openai dep dependency if not installed, or use standard if available.
# We'll try standard imports assuming user fixes env.

from project.llm_utils import call_llm

# --- Config ---
DATA_DIR = "project/data_processed"
CACHE_FILE = f"{DATA_DIR}/intent_cache.pkl"

# Removed local call_gpt_4o_mini function as we now use shared util


from project.config import INTENT_BATCH_SIZE

def process_intents():
    print("Loading data for intent inference...")
    try:
        with open(f"{DATA_DIR}/train.pkl", 'rb') as f:
            train_data = pickle.load(f)
        with open(f"{DATA_DIR}/test.pkl", 'rb') as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        print("Data files not found. Run data_processor.py first.")
        return

    intent_map = {} 
    all_samples = train_data + test_data
    print(f"Processing {len(all_samples)} samples in batches of {INTENT_BATCH_SIZE}...")
    
    # Filter out already processed
    to_process = []
    for s in all_samples:
        uid = s['user_id']
        ts_str = str(s['target']['Local_sg_time'])
        # If we had a persistent cache load here we could skip
        # For now assume clean run or handled by map update
        to_process.append((uid, ts_str, s))

    # Batch Process
    for i in range(0, len(to_process), INTENT_BATCH_SIZE):
        batch = to_process[i:i+INTENT_BATCH_SIZE]
        if not batch: break
        
        # 1. Prepare Batch Prompt
        batch_input = []
        for uid, ts_str, s in batch:
            # Shorten history
            hist_parts = []
            for h in s['history'][-3:]: # Keep context very short for batching
                cat = h.get('Category', 'Place')
                t = h.get('TimeSlot', 'Day')
                hist_parts.append(f"{cat}@{t}")
            hist_str = ", ".join(hist_parts)
            curr_t = s['target'].get('TimeSlot', 'Unknown')
            
            batch_input.append({
                "id": f"{uid}_{ts_str}", # Unique key for mapping back
                "history": hist_str,
                "time": curr_t
            })
            
        prompt = (
            f"Infer intent (1 word) for these users.\n"
            f"Input: {json.dumps(batch_input)}\n"
            f"Output JSON: {{ \"results\": [ {{ \"id\": \"uid_ts\", \"intent\": \"...\" }}, ... ] }}"
        )
        
        # 2. Call LLM
        response = call_llm(prompt, system_role="Batch Intent Inference. Output valid JSON only.")
        
        # 3. Parse & Store
        try:
            if response:
                # Cleanup potential code blocks
                clean_res = response.replace("```json", "").replace("```", "").strip()
                match = json.loads(clean_res)
                results = match.get("results", [])
                
                # Update Map
                res_map = {r['id']: r['intent'] for r in results}
                
                for uid, ts_str, _ in batch:
                    key = f"{uid}_{ts_str}"
                    intent = res_map.get(key, "General") # Fallback if missing in output
                    intent_map[(uid, ts_str)] = intent
            else:
                # Fallback for entire batch failure
                 for uid, ts_str, _ in batch:
                    intent_map[(uid, ts_str)] = "mock_intent_batch_fail"
                    
        except json.JSONDecodeError:
            print(f"Batch {i} JSON Fail.")
            if response:
                print(f"  Response Snippet: {response[:200]}...")
                print(f"  Response Length: {len(response)}")
            else:
                print("  Response was empty/None.")
            
            for uid, ts_str, _ in batch:
                intent_map[(uid, ts_str)] = "mock_intent_json_fail"
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(to_process)} samples...")

    print(f"Saving {len(intent_map)} intents to {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(intent_map, f)

if __name__ == "__main__":
    process_intents()
