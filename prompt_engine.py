import json
from project.config import MAX_PROMPT_TOKENS

class PromptEngine:
    def __init__(self, max_tokens=MAX_PROMPT_TOKENS): # Increased for safety
        self.max_tokens = max_tokens
        
    def build_prompt_and_check(self, user_ctx, candidates, evidence_cards, action):
        """
        Constructs and checks prompt length.
        action: dict from ActionSpace {'M':.., 'style':.., 'w':.., 'order':..}
        candidates: List of dicts [{'id':.., 'cat':.., 'dist':..}, ...]
        """
        # 1. System Prompt
        sys_p = f"User is in {user_ctx['time_slot']}, looking for {user_ctx['intent']}..."
        
        # 2. Candidates
        cand_str = json.dumps(candidates) 
        
        # 3. Evidence Cards
        cards_str = self.format_cards(evidence_cards, action['M'], action['style'], action['order'])
        
        full_text = f"{sys_p}\nCandidates: {cand_str}\nReasons:\n{cards_str}\n\nTask:\nRank candidates based on likelihood of visit.\nOutput JSON: {{\"ranking\": [ID1, ID2, ...]}}"
        
        # 4. Token Check
        est_tokens = self.count_tokens(full_text)
        if est_tokens > self.max_tokens:
            print(f" [PromptEngine] Limit Exceeded: {int(est_tokens)} > {self.max_tokens}")
            return None
            
        return full_text

    def format_cards(self, evidence_map, M, style, order):
        text = ""
        import random
        
        for cid, paths in evidence_map.items():
            text += f"Candidate {cid}:\n"
            
            # Apply Ordering (Alpha)
            if order == 'diversity':
                # Simple shuffle to simulate diversity prioritization
                # Real diversity would require embedding checks
                selected_pool = list(paths)
                random.shuffle(selected_pool)
                selected = selected_pool[:M]
            else:
                # score_desc (Default from miner)
                selected = paths[:M]
            
            if not selected:
                text += "  (No info)\n"
                continue
            
            for i, p in enumerate(selected):
                if style == 'detailed': 
                   text += f"  - Reason {i+1}: {p}\n"
                else:
                   text += f"  - {p}\n"
        return text

    def count_tokens(self, text):
        # Heuristic: 1 token ~= 4 chars
        return len(text) / 4.0

    # Legacy method wrapper if needed, but we replace construct_prompt
    def construct_prompt(self, context, candidates, evidence_map, action_tuple):
        # backward compatibility or helper
        # action_tuple was (M, style, w, order)
        # Adapt to dict
        act = {'M': action_tuple[0], 'style': action_tuple[1], 'w': action_tuple[2], 'order': action_tuple[3]}
        return self.build_prompt_and_check(context, candidates, evidence_map, act)
