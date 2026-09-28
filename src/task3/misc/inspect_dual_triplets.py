import json
import random
from pathlib import Path

def inspect_dual_modality():
    # Point to the new Phase 3 dataset
    path = Path("data/processed/phase3_hard_negatives.jsonl")
    
    if not path.exists():
        print(f"Error: Could not find {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} dual-modality triplets. Drawing 3 random samples...\n")
    print("="*80)
    
    samples = random.sample(lines, 3)
    for i, line in enumerate(samples):
        entry = json.loads(line)
        
        # Truncate to 120 characters to keep terminal clean, 
        # but long enough to verify the XML tags.
        trunc = 120
        
        print(f"Sample {i+1} | Topic: {entry['topic_id']}")
        
        print("\n  --- OPT (Semantic Logic) ---")
        print(f"  [Q] Query : {str(entry['query_opt'])[:trunc]}...")
        print(f"  [+] True  : {str(entry['pos_opt'])[:trunc]}...")
        print(f"  [-] Hard  : {str(entry['hard_neg_opts'][0])[:trunc]}...")
        
        print("\n  --- SLT (Visual Layout) ---")
        print(f"  [Q] Query : {str(entry['query_slt'])[:trunc]}...")
        print(f"  [+] True  : {str(entry['pos_slt'])[:trunc]}...")
        print(f"  [-] Hard  : {str(entry['hard_neg_slts'][0])[:trunc]}...")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect_dual_modality()