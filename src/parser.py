import re
import unicodedata


FURNITURE_ALIASES = {
    "armchair": ("armchair",),
    "chair": ("dining chair", "wooden chair", "chair", "ghe", "ghế"),
    "table": ("dining table", "coffee table", "wooden table", "table", "ban", "bàn"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed", "giuong", "giường"),
    "desk": ("desk",),
    "stool": ("stool",),
    "bench": ("bench",),
    "cabinet": ("cabinet", "tu", "tủ"),
    "wardrobe": ("wardrobe",),
    "bookshelf": ("bookshelf",),
    "shelf": ("shelf",),
    "lamp": ("lamp", "den", "đèn"),
    "ottoman": ("ottoman",),
    "nightstand": ("nightstand", "bedside table"),
    "dresser": ("dresser",),
}


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text).lower())
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", without_marks).strip()


def _generic_label(prompt: str) -> str | None:
    """Keep a non-furniture object instead of collapsing it to 'furniture'."""
    text = _normalized(prompt)
    first_clause = re.split(
        r"\s*(?:,|;|\band\b|\bnext to\b|\bbeside\b|\balongside\b|\bnear\b)\s*",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    words = re.findall(r"[a-z0-9]+", first_clause)
    stopwords = {
        "a", "an", "one", "the", "exactly", "detailed", "realistic",
        "vintage", "modern", "wooden", "metal", "red", "blue", "green",
        "black", "white", "small", "large", "product", "photography",
        "isolated", "studio", "photo", "image", "object",
    }
    candidates = [word for word in words if word not in stopwords]
    return candidates[-1] if candidates else None


def _extract_labels(prompt: str) -> list[str]:
    text = _normalized(prompt)
    matches = []
    for label, aliases in FURNITURE_ALIASES.items():
        for alias in aliases:
            alias_normalized = _normalized(alias)
            for match in re.finditer(
                rf"(?<!\w){re.escape(alias_normalized)}(?!\w)", text
            ):
                matches.append((match.start(), match.end(), -len(alias_normalized), label))

    matches.sort()
    labels = []
    occupied = []
    for start, end, _, label in matches:
        if any(not (end <= old_start or start >= old_end) for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        labels.append(label)

    # Keep repeated objects. The downstream generator gives each one a unique id.
    if labels:
        return labels

    # The pipeline also supports arbitrary image-to-3D objects. Do not
    # silently relabel a bicycle, toy, vehicle, etc. as generic furniture.
    generic = _generic_label(prompt)
    return [generic or "object"]


def _relation(prompt: str) -> str:
    text = f" {_normalized(prompt)} "
    patterns = (
        ("on_top_of", (" on top of ", " placed on ", " tren ", " dat tren ")),
        ("under", (" under ", " underneath ", " below ", " duoi ")),
        ("in_front_of", (" in front of ", " phia truoc ")),
        ("behind", (" behind ", " phia sau ")),
        ("left_of", (" to the left of ", " left of ", " ben trai ")),
        ("right_of", (" to the right of ", " right of ", " ben phai ")),
        ("next_to", (" next to ", " beside ", " alongside ", " near ", " ke ben ", " ben canh ")),
    )
    for relation, phrases in patterns:
        if any(phrase in text for phrase in phrases):
            return relation
    return "single" if len(_extract_labels(prompt)) == 1 else "next_to"


def parse_scene_graph(prompt: str) -> dict:
    """Create a deterministic object-wise plan for the pipeline."""
    clean_prompt = " ".join(str(prompt).split()).strip()
    if not clean_prompt:
        raise ValueError("Prompt must not be empty")

    labels = _extract_labels(clean_prompt)
    nodes = []
    for index, label in enumerate(labels, start=1):
        nodes.append(
            {
                "id": f"{label}_{index}",
                "label": label,
                "full": f"one {label} requested by the user",
                "description": clean_prompt,
            }
        )

    relation = _relation(clean_prompt)
    edges = []
    if len(nodes) >= 2:
        edges.append(
            {
                "subject": nodes[0]["id"],
                "relation": relation,
                "object": nodes[1]["id"],
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "relation": relation,
        "raw": clean_prompt,
        "mode": "objectwise",
    }
