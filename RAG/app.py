from __future__ import annotations

import streamlit as st

from graph_rag_catalog import (
    answer_question,
    build_graph,
    compare_products,
    find_datasource,
    get_catalog_summary,
    get_products_by_category,
    load_rows,
)


@st.cache_resource
def get_graph_and_rows():
    source = find_datasource()
    rows = load_rows(source)
    graph = build_graph(rows)
    return graph, rows


graph, rows = get_graph_and_rows()
summary = get_catalog_summary(rows)

st.set_page_config(page_title="Product Datasheet GraphRAG", page_icon="📦", layout="wide")
st.title("Product Datasheet GraphRAG Assistant")
st.caption("Production-style question answering over the product datasheet graph.")

with st.sidebar:
    st.subheader("Catalog overview")
    st.metric("Products", len(summary["products"]))
    st.metric("Brands", len(summary["brands"]))
    st.metric("Categories", len(summary["categories"]))
    st.metric("Graph nodes", graph.number_of_nodes())
    st.metric("Graph edges", graph.number_of_edges())

    st.markdown("---")
    st.subheader("Filter by category")
    category = st.selectbox("Category", ["All"] + summary["categories"])
    if category != "All":
        products = get_products_by_category(rows, category)
        st.write("Products in this category:")
        for product in products:
            st.write("- " + product)

example_questions = [
    "Which smartphones are in the catalog?",
    "What is the cheapest product?",
    "Which products support 5G?",
    "Which laptops include Wi-Fi 6?",
    "What are the main brands?",
    "Which product has the highest battery?",
    "Which product is a smartwatch?",
    "Compare Alpha X1 and Alpha X2",
]

left_col, right_col = st.columns([2, 1])
with left_col:
    question = st.text_input("Ask a product question", value=example_questions[0])

with right_col:
    st.markdown("### Quick picks")
    for q in example_questions[:4]:
        if st.button(q, key=q):
            question = q

if st.button("Get Answer"):
    answer = answer_question(graph, question)
    st.markdown("### Retrieved Graph Context")
    st.code(answer.split("\n\nFinal Answer:\n")[0].replace("Retrieved Graph Context:\n", ""))
    st.markdown("### Final Answer")
    st.success(answer.split("\n\nFinal Answer:\n")[1])

if "compare" in question.lower() and "and" in question.lower():
    names = [part.strip() for part in question.split("and") if part.strip()]
    if len(names) >= 2:
        comparison = compare_products(rows, names)
        st.info("Comparison:\n" + comparison)

st.subheader("Example questions")
for q in example_questions:
    st.markdown(f"- {q}")

st.subheader("Catalog snapshot")
for product in summary["products"][:10]:
    st.write("- " + product)
