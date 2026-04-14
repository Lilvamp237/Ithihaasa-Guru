import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import requests
from langgraph.graph import END, StateGraph
from rdflib import Graph

from rag_pipeline import retrieve


class ScoreState(TypedDict, total=False):
    question_id: str
    student_answer_si: str
    question_obj: Dict[str, Any]
    marking_guide: Dict[str, Any]
    keywords: List[str]
    rag_hits: List[Dict[str, Any]]
    ontology_facts: List[str]
    grading_result: Dict[str, Any]
    final_output: Dict[str, Any]
    grading_retries: int


def load_question_from_config(state: ScoreState) -> ScoreState:
    config = json.loads(Path("history_config.json").read_text(encoding="utf-8"))
    qid = state["question_id"]
    question = next((q for q in config["questions"] if q["id"] == qid), None)
    if not question:
        raise ValueError(f"Question id not found: {qid}")
    return {
        **state,
        "question_obj": question,
        "marking_guide": question["marking_guide"],
        "keywords": question["core_keywords_si"],
    }


def ontology_context_for_question(question_obj: Dict[str, Any], ttl_path: str = "ontology_colonial.ttl") -> List[str]:
    g = Graph()
    g.parse(ttl_path, format="ttl")
    facts: List[str] = []
    qtext = question_obj["question_si"]

    if "පෘතුගීසි" in qtext or "පෘතුගීසී" in qtext:
        query = """
        PREFIX ex: <http://example.org/colonial#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?eventLabel ?year WHERE {
          ?event ex:affected_by ex:Portuguese ;
                 rdfs:label ?eventLabel ;
                 ex:event_year ?year .
          FILTER(lang(?eventLabel) = "si")
        }
        """
    elif "ලන්දේසි" in qtext:
        query = """
        PREFIX ex: <http://example.org/colonial#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?eventLabel ?year WHERE {
          ?event ex:affected_by ex:Dutch ;
                 rdfs:label ?eventLabel ;
                 ex:event_year ?year .
          FILTER(lang(?eventLabel) = "si")
        }
        """
    else:
        query = """
        PREFIX ex: <http://example.org/colonial#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?powerLabel ?nextLabel WHERE {
          ?p ex:succeeded_by ?n ;
             rdfs:label ?powerLabel .
          ?n rdfs:label ?nextLabel .
          FILTER(lang(?powerLabel)="si" && lang(?nextLabel)="si")
        }
        """

    for row in g.query(query):
        vals = [str(v) for v in row]
        facts.append(" | ".join(vals))
    return facts


def retrieval_agent(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    keywords = state["keywords"]
    query = f"{question['question_si']} {' '.join(keywords)} {state['student_answer_si']}"
    rag_hits = retrieve(query=query, out_dir=Path("vector_store"), k=5)
    onto_facts = ontology_context_for_question(question)
    return {**state, "rag_hits": rag_hits, "ontology_facts": onto_facts}


def ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def grading_agent(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    guide = state["marking_guide"]
    answer = state["student_answer_si"]
    rag_hits = state.get("rag_hits", [])
    onto_facts = state.get("ontology_facts", [])

    prompt = f"""
You are a strict Sinhala history grader.
Question (Sinhala): {question['question_si']}
Marking guide JSON: {json.dumps(guide, ensure_ascii=False)}
Student answer (Sinhala): {answer}

Retrieved textbook evidence:
{json.dumps(rag_hits, ensure_ascii=False)}

Ontology facts:
{json.dumps(onto_facts, ensure_ascii=False)}

Task:
1) Score each criterion independently using ONLY the marking guide maxima.
2) Give Sinhala explanation per criterion with evidence reference.
3) Return valid JSON only with structure:
{{
  "criteria_scores": [
    {{
      "criterion_id": "qX_cY",
      "awarded_marks": 0,
      "max_marks": 0,
      "explanation_si": "..."
    }}
  ],
  "total_score": 0,
  "final_feedback_si": "..."
}}
Ensure total_score is the exact sum and <= 20.
"""

    raw = ollama_chat(model="gemma3:12b", prompt=prompt)
    try:
        grading = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            grading = json.loads(raw[start : end + 1])
        else:
            raise ValueError("LLM output is not valid JSON.")
    return {**state, "grading_result": grading}


def grade_quality_gate(state: ScoreState) -> str:
    """Create a cyclic LangGraph loop when grading JSON is structurally invalid."""
    guide = state["marking_guide"]
    expected = len(guide["criteria"])
    retries = state.get("grading_retries", 0)
    result = state.get("grading_result", {})
    criteria = result.get("criteria_scores", [])

    if len(criteria) != expected:
        if retries < 1:
            state["grading_retries"] = retries + 1
            return "retry_grading"
        raise ValueError("Grading output criterion count mismatch.")

    computed_total = 0
    for item in criteria:
        max_marks = int(item.get("max_marks", 0))
        awarded = int(item.get("awarded_marks", 0))
        if awarded < 0 or awarded > max_marks:
            if retries < 1:
                state["grading_retries"] = retries + 1
                return "retry_grading"
            raise ValueError("Invalid awarded marks in grading output.")
        computed_total += awarded

    if computed_total != int(result.get("total_score", -1)) or computed_total > 20:
        if retries < 1:
            state["grading_retries"] = retries + 1
            return "retry_grading"
        raise ValueError("Grading output total score mismatch.")

    return "to_justification"


def justification_agent(state: ScoreState) -> ScoreState:
    grading = state["grading_result"]
    rag_hits = state.get("rag_hits", [])
    onto_facts = state.get("ontology_facts", [])

    evidence = []
    for h in rag_hits[:3]:
        evidence.append(
            {
                "source": h["source"],
                "chunk_id": h["chunk_id"],
                "snippet": h["text"][:240],
                "score": h["score"],
            }
        )

    grading["evidence_summary"] = {
        "rag_top_hits": evidence,
        "ontology_facts": onto_facts,
    }
    return {**state, "grading_result": grading}


def output_formatter(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    out = {
        "question_id": question["id"],
        "question_si": question["question_si"],
        "score_out_of_20": state["grading_result"]["total_score"],
        "criteria_breakdown": state["grading_result"]["criteria_scores"],
        "final_feedback_si": state["grading_result"]["final_feedback_si"],
        "evidence": state["grading_result"]["evidence_summary"],
    }
    return {**state, "final_output": out}


def build_graph():
    graph = StateGraph(ScoreState)
    graph.add_node("orchestrator", load_question_from_config)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("grading_agent", grading_agent)
    graph.add_node("justification_agent", justification_agent)
    graph.add_node("output_formatter", output_formatter)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "retrieval_agent")
    graph.add_edge("retrieval_agent", "grading_agent")
    graph.add_conditional_edges(
        "grading_agent",
        grade_quality_gate,
        {
            "retry_grading": "grading_agent",
            "to_justification": "justification_agent",
        },
    )
    graph.add_edge("justification_agent", "output_formatter")
    graph.add_edge("output_formatter", END)
    return graph.compile()


def run_scoring(question_id: str, student_answer_si: str) -> Dict[str, Any]:
    app = build_graph()
    initial: ScoreState = {"question_id": question_id, "student_answer_si": student_answer_si}
    final = app.invoke(initial)
    return final["final_output"]


if __name__ == "__main__":
    sample = run_scoring(
        question_id="q1",
        student_answer_si="පෘතුගීසීන් 1505දී පැමිණියේ වෙළඳාම සහ මුහුදු මාර්ග පාලනය සඳහාය...",
    )
    print(json.dumps(sample, ensure_ascii=False, indent=2))
