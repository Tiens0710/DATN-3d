import sys
import json
import subprocess

PY_PATH = "/opt/venv310/bin/python"

def parse_scene_graph(prompt: str) -> dict:
    """
    Sử dụng spaCy NLP để phân tích cú pháp Scene Graph (Node & Edge) từ prompt.
    """
    # Tránh lỗi cú pháp khi prompt chứa dấu nháy hoặc ký tự đặc biệt
    safe_text = prompt.replace('\\', '\\\\').replace('"', '\\"')
    script = f"""
import sys, json, spacy, re
sys.modules['triton'] = None  # Chặn lỗi bitsandbytes

nlp = spacy.load("en_core_web_sm")
RELATION_KEYWORDS = {{
    "on": "on_top_of", "on top of": "on_top_of", "under": "under", "below": "under",
    "next to": "next_to", "beside": "next_to", "behind": "behind", "in front of": "in_front_of",
    "above": "above", "inside": "inside", "near": "next_to",
}}

def parse(text):
    doc = nlp(text.lower())
    nodes = []
    edges = []
    for i, chunk in enumerate(doc.noun_chunks):
        root = chunk.root.text
        attrs = [t.text for t in chunk if t.pos_ == "ADJ"]
        nodes.append({{"id": f"obj_{{i}}", "label": root, "attributes": attrs, "full": chunk.text}})
    
    text_lower = text.lower()
    for phrase, rel_type in RELATION_KEYWORDS.items():
        pattern = rf"(\w+(?:\s\w+)?)\s+(?:\w+\s+)*{{re.escape(phrase)}}\s+(\w+(?:\s\w+)?)"
        for subj_text, obj_text in re.findall(pattern, text_lower):
            subj_id = next((n["id"] for n in nodes if n["label"] in subj_text or subj_text in n["full"]), None)
            obj_id  = next((n["id"] for n in nodes if n["label"] in obj_text  or obj_text  in n["full"]), None)
            if subj_id and obj_id and subj_id != obj_id:
                edges.append({{"subject": subj_id, "relation": rel_type, "object": obj_id}})
    return {{"nodes": nodes, "edges": edges, "raw": text}}

print(json.dumps(parse("{safe_text}")))
"""
    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi spaCy parser: {r.stderr}")
    return json.loads(r.stdout.strip())
