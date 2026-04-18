import json
import re
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
    semantic_verification: Dict[str, Any]
    grading_result: Dict[str, Any]
    critic_report: Dict[str, Any]
    critic_feedback: str
    final_output: Dict[str, Any]
    grading_retries: int
    critic_retries: int


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
    """Retrieves FAISS context and general ontology triples."""
    question = state["question_obj"]
    keywords = state["keywords"]
    query = f"{question['question_si']} {' '.join(keywords)} {state['student_answer_si']}"
    rag_hits = retrieve(query=query, out_dir=Path("vector_store"), k=5)
    onto_facts = ontology_context_for_question(question)
    return {
        **state,
        "rag_hits": rag_hits,
        "ontology_facts": onto_facts,
    }


def semantic_verification_agent(state: ScoreState) -> ScoreState:
    """Uses SPARQL to check if entities in the answer match ontology relationships."""
    g = Graph()
    g.parse("ontology_colonial.ttl", format="ttl")
    answer = state["student_answer_si"]
    mismatches = []
    verifications = []

    # Query for known labels and their associated powers to check for historical errors
    query = """
    PREFIX ex: <http://example.org/colonial#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label ?powerLabel WHERE {
        { ?s rdfs:label ?label . ?s ex:associated_with ?power . ?power rdfs:label ?powerLabel . }
        UNION
        { ?s rdfs:label ?label . ?s ex:implemented_by ?power . ?power rdfs:label ?powerLabel . }
        UNION
        { ?s rdfs:label ?label . ?s ex:affected_by ?power . ?power rdfs:label ?powerLabel . }
        FILTER(lang(?label) = "si" && lang(?powerLabel) = "si")
    }
    """
    
    results = g.query(query)
    powers_mentioned = [p for p in ["පෘතුගීසි", "ලන්දේසි", "බ්‍රිතාන්‍ය"] if p in answer]

    for row in results:
        entity_name = str(row.label)
        correct_power = str(row.powerLabel)
        
        if entity_name in answer:
            # If student mentions entity and a power, but they don't match, flag it
            if powers_mentioned and correct_power not in powers_mentioned:
                mismatches.append(f"සාවද්‍ය සම්බන්ධය: {entity_name} අයත් වන්නේ {correct_power} පාලනයටයි.")
            else:
                verifications.append(f"නිවැරදි සම්බන්ධය: {entity_name} සහ {correct_power} අතර තොරතුරු තහවුරුයි.")

    historical_penalty = min(4, len(mismatches))
    return {
        **state,
        "semantic_verification": {
            "mismatches": mismatches,
            "verifications": verifications,
            "status": "Found Mismatches" if mismatches else "Consistent",
            "historical_inaccuracy_penalty": historical_penalty,
            "requires_deduction": historical_penalty > 0
        }
    }


def ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def grading_agent(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    guide = state["marking_guide"]
    answer = state["student_answer_si"]
    rag_hits = state.get("rag_hits", [])
    onto_facts = state.get("ontology_facts", [])
    semantic_report = state.get("semantic_verification", {})
    reviewer_feedback = state.get("critic_feedback", "")

    # Incorporating mandatory requirement: include semantic verification in system prompt
    prompt = f"""
You are a strict Sinhala history grader.
Question (Sinhala): {question['question_si']}
Marking guide JSON: {json.dumps(guide, ensure_ascii=False)}
Student answer (Sinhala): {answer}

Retrieved textbook evidence:
{json.dumps(rag_hits, ensure_ascii=False)}

Ontology facts:
{json.dumps(onto_facts, ensure_ascii=False)}

The Ontology verification agent found these results: {state['semantic_verification']}. 
If there are mismatches, you MUST deduct marks and explain why in Sinhala.

Reviewer correction feedback (if present):
{reviewer_feedback or "N/A"}

Task:
1) Perform semantic verification FIRST using the provided verification report.
2) If semantic_verification.requires_deduction is true, you MUST deduct {semantic_report.get('historical_inaccuracy_penalty', 0)} marks from the 'historical_accuracy' or relevant criterion.
3) Use specific citations for RAG hits: [RAG rank=<n> source=<path> chunk=<id>].
4) Return ONLY valid JSON with structure:
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
        awarded = int(item.get("awarded_marks", 0))
        computed_total += awarded

    if computed_total != int(result.get("total_score", -1)) or computed_total > 20:
        if retries < 1:
            state["grading_retries"] = retries + 1
            return "retry_grading"
        raise ValueError("Grading output total score mismatch.")

    return "to_critic"


def critic_agent(state: ScoreState) -> ScoreState:
    guide = state["marking_guide"]
    result = state["grading_result"]
    rag_hits = state.get("rag_hits", [])
    expected_max_total = int(guide.get("total_marks", 20))
    criteria = result.get("criteria_scores", [])
    issues: List[str] = []

    guide_by_id = {c["criterion_id"]: int(c["marks"]) for c in guide["criteria"]}
    
    awarded_total = 0
    max_total = 0
    weak_criteria: List[str] = []
    citation_regex = re.compile(r"\[RAG\s+rank=(\d+)\s+source=(.+?)\s+chunk=(\d+)\]")
    valid_citation_keys = {(h["source"], int(h.get("chunk_id", 0))) for h in rag_hits}

    for item in criteria:
        cid = item.get("criterion_id", "")
        awarded = int(item.get("awarded_marks", 0))
        cmax = int(item.get("max_marks", 0))
        explanation = str(item.get("explanation_si", ""))
        awarded_total += awarded
        max_total += cmax

        if cid in guide_by_id and cmax != guide_by_id[cid]:
            issues.append(f"max_marks_mismatch:{cid}")

        # Check for citation grounding
        matches = citation_regex.findall(explanation)
        if not matches:
            weak_criteria.append(cid)

        if not re.search(r"[\u0D80-\u0DFF]", explanation):
            issues.append(f"non_sinhala_explanation:{cid}")

    if awarded_total != int(result.get("total_score", -1)):
        issues.append("awarded_total_mismatch")
    
    grounding_ratio = (len(criteria) - len(weak_criteria)) / max(1, len(criteria))

    reviewer_pass = len(issues) == 0
    feedback = f"Revise grading: {'; '.join(issues)}. Ensure criteria match marking guide and cite RAG evidence."
    
    report = {
        "pass": reviewer_pass,
        "issues": issues,
        "grounding_ratio": round(grounding_ratio, 3),
        "weak_criteria": weak_criteria,
    }
    return {**state, "critic_report": report, "critic_feedback": "" if reviewer_pass else feedback}


def critic_quality_gate(state: ScoreState) -> str:
    report = state.get("critic_report", {})
    if report.get("pass") or state.get("critic_retries", 0) >= 1:
        return "to_justification"
    state["critic_retries"] = state.get("critic_retries", 0) + 1
    return "retry_grading"


def justification_agent(state: ScoreState) -> ScoreState:
    """Formats evidence for the UI by mapping 'text' to 'snippet'."""
    grading = state["grading_result"]
    rag_hits = state.get("rag_hits", [])
    
    # We create a clean summary for the UI
    ui_evidence = []
    for hit in rag_hits[:3]:  # Top 3 most relevant hits
        ui_evidence.append({
            "source": hit["source"],
            "chunk_id": hit.get("chunk_id", "N/A"),
            # Map 'text' to 'snippet' and truncate for a cleaner UI view
            "snippet": hit["text"][:300] + "...", 
            "score": hit["score"]
        })

    grading["evidence_summary"] = {
        "rag_top_hits": ui_evidence,
        "ontology_facts": state.get("ontology_facts", []),
        "semantic_verification": state.get("semantic_verification", {}),
    }
    return {**state, "grading_result": grading}

def confidence_agent(state: ScoreState) -> ScoreState:
    rag_hits = state.get("rag_hits", [])
    semantic_report = state.get("semantic_verification", {})
    critic_report = state.get("critic_report", {})
    
    top_scores = [float(hit.get("score", 0)) for hit in rag_hits[:3]]
    avg_similarity = sum(top_scores) / max(1, len(top_scores))
    
    # Confidence calculation incorporating ontology consistency
    mismatch_penalty = 20 if semantic_report.get("requires_deduction") else 0
    confidence = (avg_similarity * 60) + (critic_report.get("grounding_ratio", 0) * 40) - mismatch_penalty
    confidence = round(max(0, min(100, confidence)), 1)

    confidence_payload = {
        "score": confidence,
        "level": "High" if confidence >= 75 else "Medium" if confidence >= 50 else "Low",
    }
    return {**state, "grading_result": {**state["grading_result"], "confidence": confidence_payload}}


def output_formatter(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    res = state["grading_result"]
    out = {
        "question_id": question["id"],
        "question_si": question["question_si"],
        "score_out_of_20": res["total_score"],
        "criteria_breakdown": res["criteria_scores"],
        "final_feedback_si": res["final_feedback_si"],
        "evidence": res["evidence_summary"],
        "confidence": res["confidence"],
    }
    return {**state, "final_output": out}


def build_graph():
    graph = StateGraph(ScoreState)
    
    graph.add_node("orchestrator", load_question_from_config)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("semantic_verification", semantic_verification_agent)
    graph.add_node("grading_agent", grading_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node("justification_agent", justification_agent)
    graph.add_node("confidence_agent", confidence_agent)
    graph.add_node("output_formatter", output_formatter)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "retrieval_agent")
    graph.add_edge("retrieval_agent", "semantic_verification")
    graph.add_edge("semantic_verification", "grading_agent")
    
    graph.add_conditional_edges(
        "grading_agent",
        grade_quality_gate,
        {"retry_grading": "grading_agent", "to_critic": "critic_agent"},
    )
    graph.add_conditional_edges(
        "critic_agent",
        critic_quality_gate,
        {"retry_grading": "grading_agent", "to_justification": "justification_agent"},
    )
    graph.add_edge("justification_agent", "confidence_agent")
    graph.add_edge("confidence_agent", "output_formatter")
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
