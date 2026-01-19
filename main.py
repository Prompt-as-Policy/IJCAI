import pickle
import numpy as np
import re
import json
import random
import sys
import os
import zlib

# Add project root to sys.path to fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project.knowledge_graph import KnowledgeGraph
from project.user_context import UserContext
from project.graph_utils import CandidateDiscoverer, EvidenceMiner
from project.bandit import ActionSpace, LinearThompsonSampling
from project.prompt_engine import PromptEngine
from project.llm_utils import call_llm
from project.llm_utils import call_llm
from project.config import N_EPOCHS, STATE_DIM, DATA_DIR, TOP_K_CANDIDATES, BANDIT_REGULARIZATION, BANDIT_EXPLORATION

# --- Helper: State Encoding ---
def encode_user_state(ctx):
    """
    Encodes User Context (20 dim)
    TimeSlot (4) + IntentHash (8) + ProfileHash (8)
    """
    # 1. TimeSlot (One-Hot 4)
    slots = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_vec = np.zeros(4)
    if ctx['time_slot'] in slots:
        time_vec[slots.index(ctx['time_slot'])] = 1.0
        
    # 2. Intent (Hash 8)
    intent_vec = np.zeros(8)
    if ctx['intent']:
        h = zlib.adler32(ctx['intent'].encode()) % 8
        intent_vec[h] = 1.0
        
    # 3. Profile (Hash 8)
    prof_vec = np.zeros(8)
    # Profile is a string (Category Label)
    if ctx['profile'] and isinstance(ctx['profile'], str):
        h = zlib.adler32(ctx['profile'].encode()) % 8
        prof_vec[h] = 1.0
        
    return np.concatenate([time_vec, intent_vec, prof_vec])



def parse_llm_output(output_text, candidate_ids):
    try:
        match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if not match: return None
        json_str = match.group(0)
        data = json.loads(json_str)
        ranking = data.get('ranking', [])
        if not isinstance(ranking, list): return None
        
        cand_set = set(candidate_ids)
        rank_set = set(ranking)
        
        # Hallucination Penalty: All predicted IDs must be in candidate set
        if not rank_set.issubset(cand_set):
            return None
            
        return ranking
    except:
        return None

def mock_llm_call(prompt, candidates, truth):
    """
    Simulates LLM response. 
    If API Key is set in llm_utils, we can try a real call.
    Otherwise return a simulated JSON ranking putting truth at top with some noise.
    """
    # 1. Try Real Call
    response = call_llm(prompt, system_role="You are a recommender system. Output JSON only.")
    if response:
        return response

    # 2. Fallback Mock
    # Create a ranking list
    ranking = list(candidates)
    if truth in ranking:
        ranking.remove(truth)
        
    random.shuffle(ranking)
    # 80% chance truth is at index 0
    if random.random() < 0.8:
        ranking.insert(0, truth)
    else:
        rank = random.randint(1, min(len(ranking), 5))
        ranking.insert(rank, truth)
        
    return json.dumps({"ranking": ranking})

def calc_reward(ranking, ground_truth):
    if not ranking: return 0.0
    
    # NDCG@5
    try:
        # Find index (0-based)
        rank_idx = ranking.index(ground_truth)
        if rank_idx < 5:
            # Formula: 1 / log2(rank + 2), where rank is 1-based
            # rank_idx is 0-based, so rank = rank_idx + 1
            # log2(rank + 1) -> log2(rank_idx + 2)
            return 1.0 / np.log2(rank_idx + 2)
    except ValueError:
        # ground_truth not in ranking
        pass
        
    return 0.0

def main():
    print("Initializing modules...")
    kg = KnowledgeGraph(DATA_DIR)
    try:
        kg.load_data()
        kg.build_graph()
    except:
        print("KG Build failed (likely no data). Exiting.")
        return

    user_ctx = UserContext(kg)
    discoverer = CandidateDiscoverer(kg)
    miner = EvidenceMiner(kg)
    
    # State dim: Needs to match feature vector construction.
    # In reproduction, we use a simple placeholder or actual OneHot.
    # Let's define a fixed dimension for now.
    # STATE_DIM = 20 (Imported)
    
    # Initialize Action Space and Bandit
    action_space = ActionSpace()
    # UPDATED: Use LinearThompsonSampling (Single Model)
    bandit = LinearThompsonSampling(
        action_space=action_space, 
        state_dim=STATE_DIM,
        lambda_reg=BANDIT_REGULARIZATION, 
        nu=BANDIT_EXPLORATION
    )
    
    prompt_engine = PromptEngine()
    
    print("Starting Training Loop...")
    with open(f"{DATA_DIR}/train.pkl", 'rb') as f:
        train_data = pickle.load(f)
        
    for epoch in range(N_EPOCHS):
        total_reward = 0
        for i, sample in enumerate(train_data):
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i} processing...")
                
            uid = sample['user_id']
            truth = sample['target']['VenueId']
            
            # Use actual time from sample target for consistency with intent cache
            current_time = sample['target'].get('Local_sg_time')
            if not current_time:
                 from datetime import datetime
                 current_time = datetime(2012, 4, 3, 10, 0)
            
            # 1. Context
            ctx = user_ctx.get_context(uid, current_time, sample['history'])
            
            # 2. Candidate Discovery
            candidates = discoverer.get_candidates(uid, ctx['last_poi'], inferred_intent=ctx['intent'])
            if not candidates: 
                if i % 10 == 0: print(f"  Step {i}: No candidates found.")
                continue
            
            # Simulated teacher forcing for training
            if truth not in candidates:
                candidates.append(truth)
            
            # CRITICAL FIX: Cap candidate set size to avoid Context Length Explosion
            # Sort by distance to last_poi and keep top 20
            last_node = kg.graph.nodes.get(ctx['last_poi'])
            if last_node:
                 def get_dist(cid):
                     cn = kg.graph.nodes.get(cid)
                     if not cn: return 99999
                     return ((cn['lat']-last_node['lat'])**2 + (cn['lon']-last_node['lon'])**2)**0.5
                 
                 candidates.sort(key=get_dist)
                 
                 # Ensure truth is kept if we sliced it off (though sorting likely keeps it if close)
                 # But simplistic: Just take top 20. If truth is far, we might lose it for "Acc@1" reward,
                 # but that's the retrieval's fault. For training stability, let's force truth in top 20.
                 sliced = candidates[:TOP_K_CANDIDATES]
                 if truth not in sliced:
                     sliced.pop()
                     sliced.append(truth)
                 candidates = sliced
            else:
                 candidates = candidates[:TOP_K_CANDIDATES] # Fallback
            
            # Enrich candidates with Metadata for Prompt (ID, Category, Distance)
            rich_candidates = []
            last_node = kg.graph.nodes.get(ctx['last_poi'])
            
            for cid in candidates:
                c_node = kg.graph.nodes.get(cid)
                dist = 0.0
                cat = "Unknown"
                
                if c_node and last_node:
                    # dist logic matches graph_utils roughly
                    d = ((c_node['lat']-last_node['lat'])**2 + (c_node['lon']-last_node['lon'])**2)**0.5 * 111
                    dist = round(d, 2)
                    
                # Find Category
                # In build_graph we added edges 'inCategory'. find neighbors of type Category
                cats = [n for n in kg.graph.neighbors(cid) if kg.graph.nodes[n].get('type') == 'Category']
                if cats: cat = cats[0]
                
                rich_candidates.append({
                    "id": cid,
                    "category": cat,
                    "distance": dist
                })

            # Calculate Candidate Stats phi(C_t)
            cand_stats = discoverer.get_candidate_stats(candidates, ctx['last_poi'])
            
            # 3. Evidence Mining
            evidence_map = {}
            # Wait for action to mine evidence? 
            # Original code mined before action selection using default? 
            # No, we moved it AFTER action selection to use weights. Correct.
            
            # 4. Bandit Action
            # Encode User State
            user_feat = encode_user_state(ctx) # 20 dim
            s_t = np.concatenate([user_feat, cand_stats])
            s_t = s_t / (np.linalg.norm(s_t) + 1e-6)
            
            arm_idx = bandit.select_arm(s_t)
            action_params = action_space.get_action(arm_idx)
            
            # Now Mine Evidence with selected 'w'
            for c in candidates:
                evidence_map[c] = miner.mine_evidence(uid, c, weight_type=action_params['w'])

            # 5. Prompt & Check
            # Pass rich_candidates instead of ID list
            prompt = prompt_engine.build_prompt_and_check(ctx, rich_candidates, evidence_map, action_params)
            
            if prompt is None:
                # Token penalty
                if i % 10 == 0: print(f"  Step {i}: Token limit exceeded.")
                reward = 0.0
                bandit.update(arm_idx, s_t, reward)
                continue
                
            # 6. LLM Call
            # Pass candidates IDs for parsing check
            llm_output = mock_llm_call(prompt, candidates, truth)
            
            # 7. Parse & Reward
            ranking = parse_llm_output(llm_output, candidates)
            if ranking is None:
                reward = 0.0
            else:
                reward = calc_reward(ranking, truth)
                
            # 8. Update
            bandit.update(arm_idx, s_t, reward)
            total_reward += reward
            
            if i % 10 == 0:
                print(f"  Step {i} Reward: {reward}")

    print("Training finished.")
    bandit.save_params(f"{DATA_DIR}/bandit_model.pkl")
    print(f"Model saved to {DATA_DIR}/bandit_model.pkl")

if __name__ == "__main__":
    main()
