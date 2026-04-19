# 📜 ඉතිහාස ගුරු (Ithihaasa Guru)

**Ithihaasa Guru** is an advanced AI-driven educational platform developed to automate the scoring and feedback process for open-ended questions on **Sri Lankan Colonial History**. By integrating Large Language Models (LLMs) with structured knowledge graphs and textbook retrieval, the system provides students with accurate, evidence-based evaluations in the Sinhala medium.

---

## ✨ Key Features

* **Hybrid Evaluation Model**: Combines the generative power of **Ollama (gemma3:12b)** with a formal **OWL Ontology** to ensure historical factual consistency.
* **Evidence-Based Scoring (RAG)**: Uses **Retrieval-Augmented Generation** to query Grade 8–11 History textbooks, ensuring every mark awarded is backed by a specific source.
* **Stateful Workflow**: Built on **LangGraph**, the system manages complex scoring logic, from initial keyword analysis to final feedback generation.
* **Modern Heritage UI**: A modern Streamlit interface styled with **Sri Lankan heritage colors** (Maroon and Orange) and intuitive WhatsApp-green action elements.
* **Inclusive Feedback**: Provides detailed Sinhala feedback, helping students understand their "Confidence Score" and identifying specific areas for improvement.

---

## 🛠️ Technical Stack

* **Orchestration**: [LangGraph](https://www.langchain.com/langgraph) for stateful agent workflows.
* **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) for high-performance textbook snippet retrieval.
* **LLM Interface**: [Ollama](https://ollama.com/) running **gemma3:12b** for local, offline processing.
* **Semantic Layer**: [RDFLib](https://github.com/RDFLib/rdflib) for querying the `ontology_colonial.ttl` knowledge graph.
* **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS for a student-friendly experience.

---

## 📂 Project Structure

```text
├── app.py                # Streamlit UI & Custom Heritage Styling
├── scorer_workflow.py    # LangGraph scoring nodes and workflow logic
├── rag_pipeline.py       # RAG logic for querying history textbook chunks
├── ontology_builder.py   # Ontology query handlers and entity mapping
├── history_config.json   # Repository of questions and marking criteria
├── ontology_colonial.ttl # Structured knowledge graph of colonial history
├── vector_store/         # FAISS index and metadata for history PDFs
└── requirements.txt      # Python dependencies (RDFLib, FAISS, LangGraph, etc.)
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed.
* The `gemma3:12b` model:
    ```bash
    ollama pull gemma3:12b
    ```

### 2. Installation
```bash
git clone [https://github.com/lilvamp237/ithihaasa-guru.git](https://github.com/lilvamp237/ithihaasa-guru.git)
cd ithihaasa-guru
pip install -r requirements.txt
```

### 3. Usage
Launch the local server:
```bash
streamlit run app.py
```
1. Select a historical question from the dropdown menu.
2. Enter the student's answer in Sinhala.
3. Click **"ලකුණු පරීක්ෂා කරන්න"** to receive the score, detailed breakdown, and historical evidence.

---

## 🤝 Contributors

* **Sumudu Ishadi Ratnayake** - *Lead Developer & Researcher*

---

## 📜 License
This project is intended for educational and academic research purposes.
```

### 💡 Implementation Notes
* **Architecture**: The README highlights the transition from a simple dashboard to a complex **LangGraph-based** system.
* **Visuals**: Mentions the **heritage styling** you requested for the `app.py` frontend.
* **Research Alignment**: Ties the project into your research on **inclusive digital infrastructure**, reflecting your expertise in the field.