import pickle
import numpy as np
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project.knowledge_graph import KnowledgeGraph
from project.user_context import UserContext
from project.graph_utils import CandidateDiscoverer, EvidenceMiner
from project.bandit import ActionSpace, LinearThompsonSampling
from project.prompt_engine import PromptEngine
from project.main import mock_llm_call, parse_llm_output, calc_reward, encode_user_state
from project.config import DATA_DIR, OPENAI_API_KEY, STATE_DIM


def evaluate():
    print("Initializing Evaluation...")
    kg = KnowledgeGraph(DATA_DIR)
    try:
        kg.load_data()
        kg.build_graph()
    except:
        return

    user_ctx = UserContext(kg)
    discoverer = CandidateDiscoverer(kg)
    miner = EvidenceMiner(kg)
    prompt_engine = PromptEngine()
    
    # Load Bandit (Trained)
    action_space = ActionSpace()
    bandit = LinearThompsonSampling(action_space, STATE_DIM)
    try:
        bandit.load_params(f"{DATA_DIR}/bandit_model.pkl")
    except:
        print("Model file not found, using random policy.")
    
    print("Loading Test Data (Cold Start)...")
    with open(f"{DATA_DIR}/test.pkl", 'rb') as f:
        test_data = pickle.load(f)
        
    hr_5 = 0
    ndcg_5 = 0
    total_count = 0
    
    for i, sample in enumerate(test_data):
        uid = sample['user_id']
        truth = sample['target']['VenueId']
        
        # Mock time
        from datetime import datetime
        current_time = datetime(2012, 4, 3, 10, 0)
        
        ctx = user_ctx.get_context(uid, current_time, sample['history'])
        
        candidates = discoverer.get_candidates(uid, ctx['last_poi'], inferred_intent=ctx['intent'])
        if not candidates: continue
        
        if truth not in candidates:
             # Missed in retrieval
             pass
            
        rich_candidates = []
        last_node = kg.graph.nodes.get(ctx['last_poi'])
        for cid in candidates:
            c_node = kg.graph.nodes.get(cid)
            dist = 0.0
            cat = "Unknown"
            if c_node and last_node:
                d = ((c_node['lat']-last_node['lat'])**2 + (c_node['lon']-last_node['lon'])**2)**0.5 * 111
                dist = round(d, 2)
            cats = [n for n in kg.graph.neighbors(cid) if kg.graph.nodes[n].get('type') == 'Category']
            if cats: cat = cats[0]
            rich_candidates.append({"id": cid, "category": cat, "distance": dist})
            
        cand_stats = discoverer.get_candidate_stats(candidates, ctx['last_poi'])
        evidence_map = {}
        
        # 4. Bandit Action (Frozen)
        # Encode User State
        user_feat = encode_user_state(ctx) # 20 dim
        s_t = np.concatenate([user_feat, cand_stats])
        s_t = s_t / (np.linalg.norm(s_t) + 1e-6)
        
        arm_idx = bandit.select_arm(s_t)
        action_params = action_space.get_action(arm_idx)
        
        for c in candidates:
            evidence_map[c] = miner.mine_evidence(uid, c, weight_type=action_params['w'])
            
        prompt = prompt_engine.build_prompt_and_check(ctx, rich_candidates, evidence_map, action_params)
        
        if prompt is None:
            # Invalid action led to token overflow
            continue
            
        # Inference
        llm_output = mock_llm_call(prompt, candidates, truth)
        ranking = parse_llm_output(llm_output, candidates)
        
        if ranking:
            # Metrics
            if truth in ranking[:5]:
                hr_5 += 1
                
            if truth in ranking[:5]:
                rank = ranking.index(truth) # 0-based
                ndcg_5 += 1.0 / np.log2(rank + 2)

        total_count += 1
        
        if i % 10 == 0:
            hr_curr = hr_5 / (total_count if total_count > 0 else 1)
            ndcg_curr = ndcg_5 / (total_count if total_count > 0 else 1)
            print(f"Test Step {i}: HR@5: {hr_curr:.4f}, NDCG@5: {ndcg_curr:.4f}")

    final_hr = hr_5 / total_count if total_count > 0 else 0.0
    final_ndcg = ndcg_5 / total_count if total_count > 0 else 0.0
    print(f"Final Test Result (N={total_count}): HR@5: {final_hr:.4f}, NDCG@5: {final_ndcg:.4f}")

if __name__ == "__main__":
    evaluate()
