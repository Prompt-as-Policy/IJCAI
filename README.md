# Prompt-as-Policy over Knowledge Graphs for Cold-start Next POI Recommendation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper**: *Do We Really Need SFT? Prompt-as-Policy over Knowledge Graphs for Cold-start Next POI Recommendation*  
> **Conference**: IJCAI 2026 (Submitted)

## 📖 Overview

This repository contains the official implementation of **Prompt-as-Policy**, a novel framework that addresses cold-start next POI (Point-of-Interest) recommendation without requiring Supervised Fine-Tuning (SFT). 

Our approach leverages:
- **Knowledge Graphs** for structured representation of user-POI interactions
- **Contextual Bandits** (Thompson Sampling) for adaptive policy learning
- **Large Language Models** as zero-shot rankers with dynamic evidence-based prompting

## 🏗️ Architecture

The framework consists of five core modules:

1. **Data Processing & Knowledge Graph Construction** (`data_processor.py`, `knowledge_graph.py`)
   - Heterogeneous graph with User, POI, Category, Grid, Time, Intent, and Profile nodes
   - Spatial-temporal relationship modeling

2. **Offline Intent Inference** (`offline_intent.py`)
   - LLM-based user intent extraction from historical trajectories

3. **Policy Learning** (`bandit.py`)
   - Linear Thompson Sampling for contextual bandit optimization
   - Action space: Evidence weighting strategies

4. **Dynamic Prompting** (`graph_utils.py`, `prompt_engine.py`)
   - Candidate discovery via graph traversal
   - Evidence mining with semantic path verbalization
   - Token-aware prompt construction

5. **Evaluation** (`evaluate.py`)
   - Cold-start test set validation
   - Metrics: Accuracy@1, NDCG@5

## 📁 Project Structure

```
Prompt-as-Policy/
├── README.md
├── requirements.txt
├── GITHUB_UPLOAD_GUIDE.md
└── project/
    ├── config.py                 # Configuration and hyperparameters
    ├── data_processor.py         # Data preprocessing
    ├── knowledge_graph.py        # KG construction
    ├── offline_intent.py         # Intent inference
    ├── user_context.py           # User context encoding
    ├── bandit.py                 # Thompson Sampling bandit
    ├── graph_utils.py            # Candidate discovery & evidence mining
    ├── prompt_engine.py          # Prompt construction
    ├── llm_utils.py              # LLM API wrapper (OpenAI/Google)
    ├── main.py                   # Training loop
    ├── evaluate.py               # Evaluation script
    ├── run_pipeline.ps1          # Windows automation script
    └── data_processed/           # Generated data (after preprocessing)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LLM API Key (OpenAI or Google Gemini)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Prompt-as-Policy/IJCAI.git
cd IJCAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API credentials:

Edit `project/config.py` or set environment variables:
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Google Gemini
export GOOGLE_API_KEY="your-api-key-here"
```

### Running the Pipeline

#### Step 1: Data Preprocessing
```bash
python project/data_processor.py
```
*Generates `train.pkl`, `test.pkl`, `poi_meta.pkl` in `project/data_processed/`*

#### Step 2: Offline Intent Inference
```bash
python project/offline_intent.py
```
*Generates `intent_cache.pkl` for user intent mapping*

#### Step 3: Policy Training
```bash
python project/main.py
```
*Trains the contextual bandit policy using Thompson Sampling*

#### Step 4: Evaluation
```bash
python project/evaluate.py
```
*Evaluates on cold-start test set and reports metrics*

### Automated Pipeline (Windows)
```powershell
.\project\run_pipeline.ps1
```

## ⚙️ Configuration

Key hyperparameters in `project/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_PROVIDER` | `"google"` | LLM provider: `"openai"` or `"google"` |
| `MODEL_NAME` | `"gemini-2.5-flash"` | Model identifier |
| `N_EPOCHS` | `1` | Training epochs |
| `STATE_DIM` | `24` | State feature dimension (20 user + 4 candidate stats) |
| `TOP_K_CANDIDATES` | `20` | Max candidates per query |
| `BANDIT_REGULARIZATION` | `1.0` | Thompson Sampling λ (regularization) |
| `BANDIT_EXPLORATION` | `0.1` | Thompson Sampling ν (exploration) |
| `MAX_PROMPT_TOKENS` | `16384` | Token limit for prompts |

## 📊 Dataset

The implementation uses the **Foursquare Calgary** dataset:
- Check-in data with spatial-temporal information
- POI categories and metadata
- Train/test split for cold-start evaluation

Data files:
- `Calgary_checkin_L1_Category.csv` - Raw check-in data
- `testdata.csv` - Test set for evaluation

## 🙏 Acknowledgments

- Foursquare for the Calgary check-in dataset
- OpenAI and Google for LLM APIs
- NetworkX community for graph processing tools

