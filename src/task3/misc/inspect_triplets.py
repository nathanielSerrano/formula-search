import json
import random

def inspect():
    path = "data/processed/self_mined_hard_negatives.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} triplets. Drawing 3 random samples...\n")
    print("="*60)
    
    samples = random.sample(lines, 3)
    for i, line in enumerate(samples):
        entry = json.loads(line)
        
        # We truncate the OPT string just to see the root structure
        q_str = str(entry["query_opt"]) + "..."
        p_str = str(entry["pos_opt"]) + "..."
        hn_str = str(entry["hard_neg_opts"][0]) + "..."
        
        print(f"Sample {i+1} | Topic: {entry['topic_id']}")
        print(f"  [Q] Query : {q_str}")
        print(f"  [+] True  : {p_str}")
        print(f"  [-] Hard  : {hn_str}")
        print("-" * 60)

if __name__ == "__main__":
    inspect()