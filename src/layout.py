def compute_layout(scene_graph: dict) -> dict:
    """
    Tính toán tọa độ Bounding Box 2D cho các đối tượng dựa trên Scene Graph.
    """
    nodes = scene_graph.get("nodes", [])
    edges = scene_graph.get("edges", [])
    layout = {}
    canvas = 512
    n = len(nodes)
    
    if n == 0:
        return {"layout": {}}
        
    default_w = int(canvas * 0.45)
    default_h = int(canvas * 0.45)
    
    if n == 1:
        layout[nodes[0]["id"]] = {
            "x": canvas // 2 - default_w // 2,
            "y": canvas // 2 - default_h // 2,
            "w": default_w, "h": default_h
        }
    else:
        cols = 2 if n <= 4 else 3
        cell_w = canvas // cols
        cell_h = canvas // ((n + cols - 1) // cols)
        for i, node in enumerate(nodes):
            col = i % cols
            row = i // cols
            layout[node["id"]] = {
                "x": col * cell_w + cell_w // 8,
                "y": row * cell_h + cell_h // 8,
                "w": int(cell_w * 0.75),
                "h": int(cell_h * 0.75)
            }
        
        def clamp(box):
            box["x"] = max(0, min(box["x"], canvas - box["w"]))
            box["y"] = max(0, min(box["y"], canvas - box["h"]))
            return box
            
        def place_above(s, o):
            o["x"], o["y"], o["w"], o["h"] = canvas // 4, int(canvas * 0.50), canvas // 2, int(canvas * 0.35)
            s["x"], s["y"], s["w"], s["h"] = canvas // 4, int(canvas * 0.10), canvas // 2, int(canvas * 0.35)
            return clamp(s), clamp(o)
            
        def place_beside(s, o):
            w = canvas // 2 - 16
            h = int(canvas * 0.60)
            y = (canvas - h) // 2
            s.update({"x": 8, "y": y, "w": w, "h": h})
            o.update({"x": canvas//2+8, "y": y, "w": w, "h": h})
            return s, o

        def place_behind(s, o):
            o["x"], o["y"], o["w"], o["h"] = canvas//4, int(canvas * 0.40), canvas // 2, int(canvas * 0.50)
            s["x"], s["y"], s["w"], s["h"] = int(canvas * 0.30), int(canvas * 0.10), int(canvas * 0.40), int(canvas * 0.35)
            return clamp(s), clamp(o)

        for edge in edges:
            s_id, rel, o_id = edge["subject"], edge["relation"], edge["object"]
            if s_id in layout and o_id in layout:
                if rel in ["on_top_of", "above"]:
                    layout[s_id], layout[o_id] = place_above(layout[s_id], layout[o_id])
                elif rel == "under":
                    layout[o_id], layout[s_id] = place_above(layout[o_id], layout[s_id])
                elif rel in ["next_to", "beside"]:
                    layout[s_id], layout[o_id] = place_beside(layout[s_id], layout[o_id])
                elif rel == "behind":
                    layout[s_id], layout[o_id] = place_behind(layout[s_id], layout[o_id])
                elif rel == "in_front_of":
                    layout[o_id], layout[s_id] = place_behind(layout[o_id], layout[s_id])
                    
    return {"layout": layout}
