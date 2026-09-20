# GraphRAG for Datasheet QA

This project builds a lightweight GraphRAG-style retrieval pipeline for a product datasheet. It reads the datasheet text, extracts entities and relations, stores them in a NetworkX graph, and answers questions by combining graph lookup with simple semantic retrieval.

## Folder structure

- `data/sample_datasheet.txt` — example datasheet used for the demo
- `graph_rag.py` — extraction, indexing, and query logic
- `requirements.txt` — Python dependencies

## Why this approach

Traditional RAG uses document chunks as flat text. GraphRAG adds a graph layer so retrieval can follow meaningful relationships such as:

- Product -> supports -> 5V DC power input
- Product -> includes -> Wi-Fi interface
- Product -> measures -> 45mm x 30mm x 12mm

This is especially useful for technical documents where the answer depends on the interaction of specifications, interfaces, and constraints.

## How to run

```bash
python -m pip install -r requirements.txt
python graph_rag.py
```

## Expected output

The script prints a graph summary and answers sample questions such as:

- What power input does the X200 Sensor Module require?
- What interface does it use for connectivity?
- What temperature range does it support?
- What are the dimensions and weight?

Example answer format:

```text
Q: What power input does the X200 Sensor Module require?
A: The datasheet states that X200 Sensor Module supports a 5V DC power input and requires a stable supply between 4.75V and 5.25V. Typical current draw is 80mA.
```

## To use your own datasheet

Replace `data/sample_datasheet.txt` with your real datasheet content or point `DATA_PATH` in `graph_rag.py` at your file.
