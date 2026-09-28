import json

def get_dataset_index_ranges(json_path: str, target_dataset: str, split: str = "train"):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    split_data = data.get(split, {})
    
    target_lower = target_dataset.lower()
    indices = sorted([
        int(idx) for idx, name in split_data.items() 
        if name.lower() == target_lower
    ])
    
    if not indices:
        print(f">>> Dataset '{target_dataset}' not found in '{split}'.")
        return []

    ranges = []
    start = indices[0]
    prev = indices[0]
    
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev))
            start = idx
            prev = idx
    ranges.append((start, prev))
    
    # result
    print(f"\n>>> Dataset: {target_dataset} ({split})")
    print(f">>> Num clips: {len(indices)}")
    print(f">>> Range: [{indices[0]} ... {indices[-1]}]")
    
    if len(ranges) == 1:
        print(f">>> Single range from {ranges[0][0]} to {ranges[0][1]}")
    else:
        print(f">>> {len(ranges)} subranges:")
        for r_start, r_end in ranges:
            print(f"  - from {r_start} to {r_end} ({r_end - r_start + 1})")
            
    return ranges


if __name__ == "__main__":
    JSON_PATH = "handx/handx_to_dataset_source_release.json"
    ranges = get_dataset_index_ranges(JSON_PATH, target_dataset="handx")