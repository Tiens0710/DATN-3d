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


# A noun after a spatial relation is often only a reference to an object
# already introduced earlier, for example: "one table and one chair beside
# the table". It must not become a second table node.
_RELATION_REFERENCE_RE = re.compile(
    r"(?:next to|beside|alongside|near|to the left of|to the right of|"
    r"left of|right of|ke ben|ben canh|ben trai|ben phai)\s+"
    r"(?:(?:the|this|that|it|cai|chiec|do|vat)\s+)?$",
    flags=re.IGNORECASE,
)
_EXPLICIT_OBJECT_RE = re.compile(
    r"(?:\b(?:a|an|one|another|two|three|four|exactly\s+one|"
    r"mot|hai|ba|bon)\b\s+)(?:\w+\s+){0,3}$",
    flags=re.IGNORECASE,
)


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

        # Do not count a relational reference as a new object when that
        # object has already been explicitly introduced. Keep it when the
        # relation target is the only mention, or when it has its own
        # determiner ("beside another table").
        prefix = text[max(0, start - 80):start]
        if (
            label in labels
            and _RELATION_REFERENCE_RE.search(prefix)
            and not _EXPLICIT_OBJECT_RE.search(prefix)
        ):
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


def _safe_node_id(label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{slug or 'object'}_{index}"


def _coerce_count(value: object) -> int:
    if isinstance(value, str):
        words = {
            "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8,
            "mot": 1, "hai": 2, "ba": 3, "bon": 4,
        }
        normalized = _normalized(value)
        if normalized in words:
            return words[normalized]
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def parse_scene_graph_from_objects(
    prompt: str,
    objects: list[dict],
    relation: str | None = None,
    relations: list[dict] | None = None,
    parser_source: str = "structured",
) -> dict:
    """Build the common graph format from arbitrary object specifications.

    ``objects`` may contain labels outside the furniture alias list. ``count``
    is expanded into individual nodes because each node becomes one image,
    segmentation crop, and TRELLIS mesh.
    """
    clean_prompt = " ".join(str(prompt).split()).strip()
    if not clean_prompt:
        raise ValueError("Prompt must not be empty")
    if not objects:
        raise ValueError("At least one object is required")

    nodes = []
    groups = []
    for spec in objects:
        label = re.sub(r"\s+", " ", str(spec.get("label", "object"))).strip()
        label = label.strip(" ,.;:").lower() or "object"
        description = " ".join(
            str(spec.get("description", "")).split()
        ).strip() or f"one {label} requested by the user"
        count = max(1, min(_coerce_count(spec.get("count", 1)), 8))

        group = []
        for _ in range(count):
            node_index = len(nodes) + 1
            node = {
                "id": _safe_node_id(label, node_index),
                "label": label,
                "full": description,
                "description": description,
            }
            nodes.append(node)
            group.append(node["id"])
        groups.append(group)

    valid_relations = {
        "single", "next_to", "on_top_of", "under", "in_front_of",
        "behind", "left_of", "right_of",
    }
    graph_relation = relation if relation in valid_relations else None
    edges = []

    # Gemini relations refer to object-group indexes, not expanded node ids.
    for edge in relations or []:
        try:
            subject_group = groups[int(edge["subject"])]
            object_group = groups[int(edge["object"])]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        edge_relation = str(edge.get("relation", "next_to"))
        if edge_relation not in valid_relations:
            edge_relation = "next_to"
        if len(subject_group) == len(object_group):
            pairs = zip(subject_group, object_group)
        elif len(object_group) == 1:
            pairs = ((subject, object_group[0]) for subject in subject_group)
        elif len(subject_group) == 1:
            pairs = ((subject_group[0], obj) for obj in object_group)
        else:
            pairs = (
                (subject, obj)
                for subject in subject_group
                for obj in object_group
            )
        edges.extend(
            {
                "subject": subject,
                "relation": edge_relation,
                "object": obj,
            }
            for subject, obj in pairs
            if subject != obj
        )

    if not edges and len(nodes) >= 2:
        graph_relation = graph_relation or "next_to"
        edges.extend(
            {
                "subject": nodes[index]["id"],
                "relation": graph_relation,
                "object": nodes[index + 1]["id"],
            }
            for index in range(len(nodes) - 1)
        )
    elif len(nodes) == 1:
        graph_relation = "single"
    else:
        graph_relation = graph_relation or "next_to"

    return {
        "nodes": nodes,
        "edges": edges,
        "relation": graph_relation,
        "raw": clean_prompt,
        "mode": "objectwise",
        "parser_source": parser_source,
    }


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
    return parse_scene_graph_from_objects(
        clean_prompt,
        [
            {
                "label": label,
                "count": 1,
                "description": f"one {label} requested by the user",
            }
            for label in labels
        ],
        relation=_relation(clean_prompt),
        parser_source="deterministic_fallback",
    )
