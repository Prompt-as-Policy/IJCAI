import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
CACHE_FILE = os.path.join(DATA_DIR, "intent_cache.pkl")

# --- LLM API ---
# Provider: "openai" or "google"
LLM_PROVIDER = "google" 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model Name (e.g., "gpt-4o-mini", "gemini-1.5-flash", "gemini-2.0-flash-exp")
MODEL_NAME = "gemini-2.5-flash" # User requested 2.5, arguably 2.0-flash-exp or 1.5-flash. Using 2.0-exp as latest.
LLM_REQUEST_DELAY = 7.0 # 10 RPM Limit -> 1 req every 6s. Set 7s for safety.
LLM_MAX_RETRIES = 5
LLM_RETRY_BASE_WAIT = 2.0 # Seconds
INTENT_BATCH_SIZE = 100 # Reduced to 100 for stability. 1000 fails JSON parsing.
# RPD Check: 13,000 items / 100 batch = 130 requests.
# Limit is 500 RPD. 130 < 500. Safe.

# --- Training Hyperparameters ---
N_EPOCHS = 1
STATE_DIM = 24 # 20 (User) + 4 (Candidate Stats)
TOP_K_CANDIDATES = 20  # Cap on candidates to prevent context overflow

# --- Prompt Engine ---
MAX_PROMPT_TOKENS = 16384

# --- Bandit ---
BANDIT_REGULARIZATION = 1.0
BANDIT_EXPLORATION = 0.1 # nu

# --- Graph / Candidate Discovery ---
NEAR_DISTANCE_THRESHOLD_KM = 10.0
MAX_HOPS = 3

