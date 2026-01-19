import random
import pickle
from datetime import datetime
from project.config import CACHE_FILE

class UserContext:
    def __init__(self, kg, intent_cache_path=CACHE_FILE):
        self.kg = kg
        self.profiles = {}
        self.intent_cache = {}
        try:
            with open(intent_cache_path, 'rb') as f:
                self.intent_cache = pickle.load(f)
            print(f"Loaded {len(self.intent_cache)} offline intents.")
        except:
            print("Warning: Intent Cache not found. Using fallback.")

    def get_context(self, user_id, current_time, history):
        """
        Constructs the user context vector x.
        """
        # 1. Basic Features
        time_slot = self._get_time_slot(current_time)
        last_poi = history[-1]['VenueId'] if history else None
        
        # 2. Profile Features
        if user_id not in self.profiles:
            self.profiles[user_id] = self._build_profile(user_id, history)
        profile = self.profiles[user_id]
        
        # 3. Intent Inference
        intent = self._lookup_intent(user_id, current_time)
        
        return {
            'user_id': user_id,
            'time_slot': time_slot,
            'last_poi': last_poi,
            'profile': profile,
            'intent': intent
        }

    def _get_time_slot(self, dt):
        h = dt.hour
        if 6 <= h < 12: return 'Morning'
        elif 12 <= h < 18: return 'Afternoon'
        elif 18 <= h < 24: return 'Evening'
        else: return 'Night'

    def _build_profile(self, user_id, history):
        cats = []
        for checkin in history:
            vid = checkin['VenueId']
            if vid in self.kg.graph.nodes:
                for n in self.kg.graph.neighbors(vid):
                    if self.kg.graph.nodes[n].get('type') == 'Category':
                        cats.append(n)
        
        if not cats: return "Generic"
        from collections import Counter
        top_cat = Counter(cats).most_common(1)[0][0]
        return top_cat

    def _lookup_intent(self, user_id, current_time):
        """
        Look up intent from offline cache.
        """
        ts_str = str(current_time)
        return self.intent_cache.get((user_id, ts_str), "General")
