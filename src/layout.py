def compute_layout(scene_graph: dict) -> dict:
    """Keep a small 2D preview while preserving semantic 3D placement data."""
    nodes = list(scene_graph.get("nodes", []))
    edges = list(scene_graph.get("edges", []))
    if not nodes:
        return {"layout": {}, "objects": [], "relation": "single"}

    relation = str(scene_graph.get("relation", ""))
    subject_id = None
    target_id = None
    if edges:
        relation = str(edges[0].get("relation", relation or "next_to"))
        subject_id = edges[0].get("subject")
        target_id = edges[0].get("object")
    if len(nodes) == 1:
        relation = "single"
    elif not relation:
        relation = "next_to"

    relations = [
        {
            "subject": edge.get("subject"),
            "relation": str(edge.get("relation", "next_to")),
            "object": edge.get("object"),
        }
        for edge in edges
        if edge.get("subject") and edge.get("object")
    ]

    boxes = {}
    if len(nodes) == 1:
        boxes[nodes[0]["id"]] = {"x": 128, "y": 80, "w": 256, "h": 352}
    else:
        columns = min(len(nodes), 3)
        rows = (len(nodes) + columns - 1) // columns
        cell_width = 512 // columns
        cell_height = 512 // rows
        for index, node in enumerate(nodes):
            boxes[node["id"]] = {
                "x": (index % 3) * cell_width + 16,
                "y": (index // 3) * cell_height + 16,
                "w": cell_width - 32,
                "h": cell_height - 32,
            }

    objects = [
        {
            "id": node["id"],
            "label": node["label"],
            "description": node.get("description", scene_graph.get("raw", "")),
        }
        for node in nodes
    ]
    return {
        "layout": boxes,
        "objects": objects,
        "relation": relation,
        "subject_id": subject_id or nodes[0]["id"],
        "target_id": target_id if len(nodes) > 1 else None,
        "relations": relations,
        "placements": list(scene_graph.get("placements", [])),
        "relation_source": scene_graph.get("parser_source", "deterministic_fallback"),
        "mode": "objectwise",
    }
