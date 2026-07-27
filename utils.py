import json
import os
from typing import Dict, List

def load_json(input_file: str) -> List[Dict]:
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(output_file: str, data: List[Dict]):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_prompt(input_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read()
