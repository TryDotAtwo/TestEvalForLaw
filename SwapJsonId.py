import json
from pathlib import Path
from collections import defaultdict

# Define paths
expert_eval_dir = Path("Expert_eval")
hash_map_file = Path("text_map.anonymized.jsonL")  # Updated filename

# Check if text_map.anonymized.jsonL exists
if not hash_map_file.exists():
    raise FileNotFoundError("text_map.anonymized.jsonL not found in the current directory")

# Load the HashTextMap as JSONL (line-by-line JSON objects)
hash_map = []
with open(hash_map_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                item = json.loads(line.strip())
                hash_map.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")

# Group texts by new_id (hash_id) and collect old_id (assuming one old_id per new_id)
hash_id_lookup = {}
for item in hash_map:
    new_id = item.get('new_id')
    if not new_id:
        print(f"Warning: Skipping item without 'new_id': {item}")
        continue
    
    if new_id not in hash_id_lookup:
        hash_id_lookup[new_id] = {
            'id': item.get('old_id', ''),
            'texts': []
        }
    hash_id_lookup[new_id]['texts'].append(item.get('text', ''))

# Join texts into full_text for each hash_id
for new_id, entry in hash_id_lookup.items():
    entry['full_text'] = ''.join(entry['texts'])
    del entry['texts']  # Clean up

# Define the score fields to extract
score_fields = [
    "plaintiff_claims",
    "plaintiff_arguments",
    "defendant_arguments",
    "evaluation_of_evidence",
    "intermediate_conclusions",
    "applicable_laws",
    "judgment_summary"
]

# Function to process a single JSON file
def process_json_file(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update each object in the JSON file
        updated_data = {}
        for short_id, value in data.items():
            # Check if the short_id matches any new_id in the mapping
            if short_id in hash_id_lookup:
                # Get the corresponding entry from the mapping
                hash_entry = hash_id_lookup[short_id]
                
                # Extract only the score fields from the existing value
                scores = {k: value.get(k, None) for k in score_fields if k in value}
                
                # Create updated entry with only scores, id, and full_text
                updated_entry = scores
                updated_entry['id'] = hash_entry['id']
                updated_entry['full_text'] = hash_entry['full_text']
                
                updated_data[short_id] = updated_entry
            else:
                # If no match is found, keep the original data (or skip? Here we keep for safety)
                updated_data[short_id] = value
                print(f"Warning: No matching new_id '{short_id}' found in text_map.anonymized.jsonL")
        
        # Write the updated data back to the JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=4, ensure_ascii=False)
        print(f"Updated: {json_path}")
    except Exception as e:
        print(f"Error processing {json_path}: {e}")

# Iterate through all subfolders and JSON files in Expert_eval
for subdir in expert_eval_dir.iterdir():
    if subdir.is_dir():  # Check if it's a directory
        for json_file in subdir.glob("*.json"):
            process_json_file(json_file)

print("All JSON files have been processed.")