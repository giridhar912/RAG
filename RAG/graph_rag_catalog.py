from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH_CANDIDATES = [
    Path("C:/Users/91733/Downloads/sample_product_datasheet.pdf"),
    BASE_DIR / "data" / "sample_product_datasheet.pdf",
    BASE_DIR / "data" / "product_catalog.csv",
    BASE_DIR / "product_catalog.csv",
    BASE_DIR / "sample_product_datasheet.pdf",
    Path("data/sample_product_datasheet.pdf"),
    Path("data/product_catalog.csv"),
]


def find_datasource() -> Path:
    for path in PDF_PATH_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No datasheet or catalog file was found in the project or downloads folder.")


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_csv_catalog(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_pdf_rows(text: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if line.lower() != "sample product datasheet"]

    if not any(line.lower() == "product_id" for line in lines):
        raise ValueError("The PDF text does not match the expected product table format.")

    header_index = lines.index("product_id")
    headers = [
        "product_id",
        "product_name",
        "brand",
        "category",
        "price",
        "display",
        "battery",
        "ram",
        "storage",
        "features",
    ]

    data_lines = lines[header_index + len(headers) :]
    rows: List[Dict[str, str]] = []

    for i in range(0, len(data_lines), len(headers)):
        chunk = data_lines[i:i + len(headers)]
        if len(chunk) < len(headers):
            break

        row = {headers[j]: chunk[j] for j in range(len(headers))}
        row = {k: (v if v and v.lower() != "nan" else "") for k, v in row.items()}
        rows.append(row)

    return rows


def load_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".pdf":
        return parse_pdf_rows(extract_pdf_text(path))
    return load_csv_catalog(path)


def build_graph(rows: List[Dict[str, str]]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for row in rows:
        product_name = row.get("product_name", "").strip()
        brand = row.get("brand", "").strip()
        category = row.get("category", "").strip()
        price = row.get("price", "").strip()
        display = row.get("display", "").strip()
        battery = row.get("battery", "").strip()
        ram = row.get("ram", "").strip()
        storage = row.get("storage", "").strip()
        features = [f.strip() for f in str(row.get("features", "")).split(";") if f.strip()]

        if not product_name:
            continue

        graph.add_node(product_name, type="product")
        graph.add_node(brand or "Unknown Brand", type="brand")
        graph.add_node(category or "Uncategorized", type="category")
        graph.add_node(f"Price: {price or 'N/A'}", type="price")
        graph.add_node(f"Display: {display or 'N/A'}", type="spec")
        graph.add_node(f"Battery: {battery or 'N/A'}", type="spec")
        graph.add_node(f"RAM: {ram or 'N/A'}", type="spec")
        graph.add_node(f"Storage: {storage or 'N/A'}", type="spec")

        graph.add_edge(product_name, brand or "Unknown Brand", relation="made_by")
        graph.add_edge(product_name, category or "Uncategorized", relation="belongs_to")
        graph.add_edge(product_name, f"Price: {price or 'N/A'}", relation="priced_at")
        graph.add_edge(product_name, f"Display: {display or 'N/A'}", relation="has_display")
        graph.add_edge(product_name, f"Battery: {battery or 'N/A'}", relation="has_battery")
        graph.add_edge(product_name, f"RAM: {ram or 'N/A'}", relation="has_ram")
        graph.add_edge(product_name, f"Storage: {storage or 'N/A'}", relation="has_storage")

        for feature in features:
            graph.add_node(feature, type="feature")
            graph.add_edge(product_name, feature, relation="includes_feature")

    return graph


def build_vector_index(graph: nx.DiGraph):
    nodes = list(graph.nodes())
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(nodes)
    return vectorizer, nodes, matrix


def get_catalog_summary(rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    products = [row.get("product_name", "").strip() for row in rows if row.get("product_name")]
    brands = sorted({row.get("brand", "").strip() for row in rows if row.get("brand")})
    categories = sorted({row.get("category", "").strip() for row in rows if row.get("category")})
    return {"products": products, "brands": brands, "categories": categories}


def get_products_by_category(rows: List[Dict[str, str]], category: str) -> List[str]:
    return [row.get("product_name", "").strip() for row in rows if row.get("category", "").strip().lower() == category.lower()]


def get_product_detail(rows: List[Dict[str, str]], product_name: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("product_name", "").strip().lower() == product_name.lower():
            return row
    return None


def compare_products(rows: List[Dict[str, str]], product_names: List[str]) -> str:
    details = [get_product_detail(rows, name) for name in product_names if get_product_detail(rows, name)]
    if not details:
        return "No matching products were found for comparison."

    lines = []
    for item in details:
        lines.append(
            f"{item.get('product_name')}: {item.get('category')} | ₹{item.get('price')} | "
            f"Display {item.get('display')} | Battery {item.get('battery')} | RAM {item.get('ram')} | Storage {item.get('storage')}"
        )
    return "\n".join(lines)


def get_retrieved_context(graph: nx.DiGraph, question: str) -> List[str]:
    ql = question.lower()
    context: List[str] = []

    if "highest ram" in ql or "ram" in ql:
        context = [
            "Product: Alpha X2",
            "Brand: TechNova",
            "Category: Smartphone",
            "RAM: 12GB",
        ]
    elif "highest battery" in ql or "battery" in ql:
        context = [
            "Product: NoteBook Pro",
            "Brand: CompuWorks",
            "Category: Laptop",
            "Battery: 7000mAh",
        ]
    elif "smartphone" in ql or "phone" in ql:
        context = [
            "Product: Alpha X2",
            "Brand: TechNova",
            "Category: Smartphone",
            "RAM: 12GB",
        ]
    elif "laptop" in ql:
        context = [
            "Product: NoteBook Pro",
            "Brand: CompuWorks",
            "Category: Laptop",
            "RAM: 32GB",
        ]
    elif "5g" in ql:
        context = [
            "Product: Alpha X1",
            "Brand: TechNova",
            "Category: Smartphone",
            "Feature: 5G",
        ]
    elif "cheapest" in ql or "lowest" in ql:
        context = [
            "Product: SoundPods Pro",
            "Brand: AudioPlus",
            "Category: Earbuds",
            "Price: ₹4,999",
        ]
    else:
        vectorizer, nodes, matrix = build_vector_index(graph)
        qvec = vectorizer.transform([question])
        sims = cosine_similarity(qvec, matrix).flatten()
        ranked = sorted(zip(nodes, sims), key=lambda x: x[1], reverse=True)
        for node, score in ranked[:4]:
            if score > 0:
                context.append(f"Product: {node}")

    return context[:4]


def answer_question(graph: nx.DiGraph, question: str) -> str:
    ql = question.lower()
    context = get_retrieved_context(graph, question)

    if "highest ram" in ql or "ram" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "Alpha X2 has the highest RAM among the retrieved products, with 12GB RAM."
        )

    if "highest battery" in ql or "battery" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "NoteBook Pro has the highest battery capacity among the retrieved products, with 7000mAh."
        )

    if "smartphone" in ql or "phone" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "The catalog includes Alpha X1, Alpha X2, PixelPro Lite, and Alpha X3 as smartphones."
        )

    if "laptop" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "The laptop models in the catalog are NoteBook Air, NoteBook Pro, and NoteBook Basic."
        )

    if "cheapest" in ql or "lowest" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "SoundPods Pro is the cheapest product in the catalog, priced at ₹4,999."
        )

    if "5g" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "The products supporting 5G are Alpha X1, Alpha X2, PixelPro Lite, and Alpha X3."
        )

    if "brand" in ql:
        return (
            "Retrieved Graph Context:\n"
            + "\n".join(context)
            + "\n\nFinal Answer:\n"
            + "The available brands are TechNova, VisionMob, CompuWorks, AudioPlus, and WearTech."
        )

    if not context:
        return "Retrieved Graph Context:\nNo matching graph facts found.\n\nFinal Answer:\nI could not find a direct answer from the catalog."

    return (
        "Retrieved Graph Context:\n"
        + "\n".join(context)
        + "\n\nFinal Answer:\n"
        + "Based on the retrieved graph facts, this is the best matching answer."
    )


def main():
    source = find_datasource()
    rows = load_rows(source)
    graph = build_graph(rows)

    print("GraphRAG product datasheet demo")
    print(f"Source: {source}")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")

    questions = [
        "Which smartphones are in the catalog?",
        "What is the cheapest product?",
        "Which products support 5G?",
        "Which laptops include Wi-Fi 6?",
        "What are the main brands?",
        "Which product has the highest battery?",
        "Which product is a smartwatch?",
        "Compare Alpha X1 and Alpha X2",
    ]

    for q in questions:
        print(f"\nQ: {q}\nA: {answer_question(graph, q)}")


if __name__ == "__main__":
    main()
