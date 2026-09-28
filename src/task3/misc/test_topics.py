import re
import sys
from pathlib import Path
import pyarrow.parquet as pq

# 1. The ACTUAL Bulletproof Path (Hardcoded to your server)
_PROJECT_ROOT = Path("/home/nathaniel.serrano/formula-rag")
_FORMULA_INDEX_DIR = _PROJECT_ROOT / "data/processed/formula_index"
_EVAL_SPLIT = "eval"

# Add root to sys.path so we can import src modules
sys.path.insert(0, str(_PROJECT_ROOT))
from src.task3.dataset import load_topics

def normalize_latex(s: str) -> str:
    """Safe normalization: whitespace, sizing, and basic synonyms only."""
    if not s:
        return ""
    # Strip ALL whitespace and newlines
    s = re.sub(r'\s+', '', s)
    # Strip visual sizing modifiers
    s = s.replace(r'\left', '').replace(r'\right', '')
    # Standardize common command synonyms
    synonyms = {
        r'\leq': r'\le', 
        r'\geq': r'\ge', 
        r'\rightarrow': r'\to', 
        r'\gets': r'\leftarrow', 
        r'\ne': r'\neq'
    }
    for old, new in synonyms.items():
        s = s.replace(old, new)
    return s

def _batch_latex_to_opt(latex_queries: set) -> dict:
    shards = sorted(_FORMULA_INDEX_DIR.glob("*.parquet"))
    
    # 2. Safety Check
    if not shards:
        print(f"\nCRITICAL ERROR: No parquet files found in {_FORMULA_INDEX_DIR}")
        sys.exit(1)

    norm_to_orig = {normalize_latex(q): q for q in latex_queries}
    normalized_query_set = set(norm_to_orig.keys())
    
    print(f"Found {len(shards)} Parquet shards. Scanning for {len(latex_queries)} normalized queries...", flush=True)
    
    results = {}
    for shard in shards:
        if len(results) == len(latex_queries):
            break 
            
        table = pq.read_table(shard, columns=["latex", "opt"])
        for lat, opt in zip(table["latex"].to_pylist(), table["opt"].to_pylist()):
            if lat:
                norm_lat = normalize_latex(lat)
                if norm_lat in normalized_query_set and opt:
                    orig_query = norm_to_orig[norm_lat]
                    if orig_query not in results:
                        results[orig_query] = opt
                        
    return results

if __name__ == "__main__":
    print("Loading topics...")
    topics = load_topics(_EVAL_SPLIT)
    unique_queries = {latex.strip() for latex in topics.values()}
    
    print(f"Found {len(unique_queries)} unique queries in the eval XML.")
    
    results = _batch_latex_to_opt(unique_queries)
    
    print(f"\nSUCCESS: Resolved {len(results)} / {len(unique_queries)} topics!")