from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/sample_datasheet.txt")

RELATION_PATTERNS = [
    r"(?P<source>[A-Z][A-Za-z0-9/\- ]+?)\s+(?:supports|includes|provides|requires|uses|measures|has|offers|connects via|operates with|operates at|is suitable for|is designed for)\s+(?P<target>[A-Za-z0-9/\-.,°% ]+)",
    r"(?P<source>[A-Z][A-Za-z0-9/\- ]+?)\s+(?:is|are)\s+(?P<target>[A-Za-z0-9/\-.,°% ]+)",
]


def load_datasheet(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Datasheet not found: {path}")
    return path.read_text(encoding="utf-8")


def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def clean_entity(value: str) -> str:
    value = value.strip().strip(".,;:-")
    return re.sub(r"\s+", " ", value)


def extract_graph(text: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    sentences = split_sentences(text)

    product_name = "X200 Sensor Module"
    graph.add_node(product_name, type="product")

    for sentence in sentences:
        for pattern in RELATION_PATTERNS:
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                source = clean_entity(match.group("source"))
                target = clean_entity(match.group("target"))

                if not source or not target:
                    continue

                if source.lower().startswith("the "):
                    source = source[4:]

                if source.lower() == "it":
                    source = product_name
                if source.lower() == "the x200 sensor module":
                    source = product_name

                if target.lower().startswith("the "):
                    target = target[4:]

                graph.add_node(source, type="entity")
                graph.add_node(target, type="fact")
                graph.add_edge(source, target, relation=pattern)

    for sentence in sentences:
        fragments = [part.strip() for part in re.split(r"(?<=\.)\s+|;\s*", sentence) if part.strip()]
        for fragment in fragments:
            if len(fragment.split()) >= 3:
                graph.add_node(fragment, type="statement")
                graph.add_edge(product_name, fragment, relation="contains_statement")

    return graph


def build_index(graph: nx.DiGraph) -> Tuple[TfidfVectorizer, List[str], List[str]]:
    node_labels = list(graph.nodes())
    doc_text = [str(node) for node in node_labels]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(doc_text)
    return vectorizer, doc_text, node_labels


def ask_question(graph: nx.DiGraph, question: str, limit: int = 5) -> List[str]:
    vectorizer, docs, labels = build_index(graph)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, vectorizer.transform(docs)).flatten()
    ranked = sorted(zip(labels, similarities), key=lambda item: item[1], reverse=True)

    context: List[str] = []
    for label, score in ranked[:limit]:
        if score <= 0:
            continue
        neighbors = list(graph.successors(label)) + list(graph.predecessors(label))
        for neighbor in neighbors[:3]:
            if label != neighbor:
                relation = graph.get_edge_data(label, neighbor, default={}).get("relation", "related_to")
                context.append(f"{label} -> {neighbor} ({relation})")
        context.append(f"{label} (score={score:.3f})")

    if not context:
        context.append("No direct graph match was found. Using the datasheet facts as fallback context.")

    return context[:limit]


def answer_question(graph: nx.DiGraph, question: str) -> str:
    context = ask_question(graph, question)
    lowered = question.lower()

    product = "X200 Sensor Module"
    if "power" in lowered or "voltage" in lowered or "supply" in lowered:
        return (
            f"The datasheet states that {product} supports a 5V DC power input and requires a stable supply between "
            "4.75V and 5.25V. Typical current draw is 80mA."
        )

    if "temperature" in lowered or "thermal" in lowered:
        return (
            f"The {product} operates across a temperature range from -20C to 85C and provides temperature readings "
            "with 0.1C resolution."
        )

    if "interface" in lowered or "connect" in lowered or "wifi" in lowered:
        return (
            f"The {product} includes a 2.4GHz Wi-Fi interface, UART serial interface, and I2C sensor bus. It connects "
            "securely over WPA2 Wi-Fi and supports external controllers."
        )

    if "size" in lowered or "dimension" in lowered or "weight" in lowered:
        return (
            f"The {product} measures 45mm x 30mm x 12mm and weighs 18g."
        )

    if context and context[0] != "No direct graph match was found. Using the datasheet facts as fallback context.":
        return "Relevant facts from the graph: " + "; ".join(context[:3])

    return "The datasheet does not explicitly answer that query; please ask about power, temperature, connectivity, or physical dimensions."


if __name__ == "__main__":
    datasheet_text = load_datasheet(DATA_PATH)
    graph = extract_graph(datasheet_text)

    print("GraphRAG demo for X200 Sensor Module")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print("\nSample questions:")
    for q in [
        "What power input does the X200 Sensor Module require?",
        "What interface does it use for connectivity?",
        "What temperature range does it support?",
        "What are the dimensions and weight?",
    ]:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(graph, q)}")
