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


def semantic_verify_with_ontology(
    question_obj: Dict[str, Any], student_answer_si: str, ttl_path: str = "ontology_colonial.ttl"
) -> Dict[str, Any]:
    g = Graph()
    g.parse(ttl_path, format="ttl")
    answer_text = student_answer_si.lower()

    event_rows = list(
        g.query(
            """
            PREFIX ex: <http://example.org/colonial#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?event ?eventLabelSi ?eventLabelEn ?power ?powerLabelSi ?powerLabelEn WHERE {
              ?event ex:affected_by ?power .
              OPTIONAL { ?event rdfs:label ?eventLabelSi . FILTER(lang(?eventLabelSi) = "si") }
              OPTIONAL { ?event rdfs:label ?eventLabelEn . FILTER(lang(?eventLabelEn) = "en") }
              OPTIONAL { ?power rdfs:label ?powerLabelSi . FILTER(lang(?powerLabelSi) = "si") }
              OPTIONAL { ?power rdfs:label ?powerLabelEn . FILTER(lang(?powerLabelEn) = "en") }
            }
            """
        )
    )
    reform_rows = list(
        g.query(
            """
            PREFIX ex: <http://example.org/colonial#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?governor ?govLabelSi ?govLabelEn ?reform ?reformLabelSi ?reformLabelEn WHERE {
              ?governor ex:implemented ?reform .
              OPTIONAL { ?governor rdfs:label ?govLabelSi . FILTER(lang(?govLabelSi) = "si") }
              OPTIONAL { ?governor rdfs:label ?govLabelEn . FILTER(lang(?govLabelEn) = "en") }
              OPTIONAL { ?reform rdfs:label ?reformLabelSi . FILTER(lang(?reformLabelSi) = "si") }
              OPTIONAL { ?reform rdfs:label ?reformLabelEn . FILTER(lang(?reformLabelEn) = "en") }
            }
            """
        )
    )

    power_aliases: Dict[str, Dict[str, Any]] = {}
    for row in event_rows:
        power_uri = str(row.power)
        label_si = str(row.powerLabelSi or "")
        label_en = str(row.powerLabelEn or "")
        aliases = {a.lower() for a in [label_si, label_en] if a}
        if "portuguese" in label_en.lower():
            aliases.update({"පෘතුගීසි", "පෘතුගීසී"})
        if "dutch" in label_en.lower():
            aliases.add("ලන්දේසි")
        if "british" in label_en.lower():
            aliases.update({"බ්‍රිතාන්‍ය", "බ්රිතාන්‍ය"})
        power_aliases[power_uri] = {"label_si": label_si, "label_en": label_en, "aliases": aliases}

    mentioned_powers: Dict[str, str] = {}
    for power_uri, meta in power_aliases.items():
        for alias in meta["aliases"]:
            if alias and alias in answer_text:
                mentioned_powers[power_uri] = meta["label_si"] or meta["label_en"] or power_uri
                break

    mismatches: List[Dict[str, str]] = []
    checked_claims: List[Dict[str, str]] = []

    for row in event_rows:
        event_uri = str(row.event)
        event_si = str(row.eventLabelSi or "")
        event_en = str(row.eventLabelEn or "")
        event_aliases = [event_si.lower(), event_en.lower()]
        if not any(alias and alias in answer_text for alias in event_aliases):
            continue

        expected_power_uri = str(row.power)
        expected_power = power_aliases.get(expected_power_uri, {}).get("label_si") or str(row.powerLabelEn or "")
        checked_claims.append(
            {
                "claim_type": "event_power",
                "entity": event_si or event_en or event_uri,
                "expected_relation": expected_power,
                "ontology_rule": f"{event_uri} ex:affected_by {expected_power_uri}",
            }
        )

        wrong_powers = [p for p in mentioned_powers.keys() if p != expected_power_uri]
        if wrong_powers:
            found_labels = [mentioned_powers[p] for p in wrong_powers]
            mismatches.append(
                {
                    "claim_type": "event_power",
                    "entity": event_si or event_en or event_uri,
                    "expected": expected_power,
                    "found": ", ".join(found_labels),
                    "ontology_rule": f"{event_uri} ex:affected_by {expected_power_uri}",
                }
            )

    known_governor_labels: Dict[str, str] = {}
    for row in reform_rows:
        gov_uri = str(row.governor)
        known_governor_labels[gov_uri] = str(row.govLabelSi or row.govLabelEn or gov_uri)

    mentioned_governors: Dict[str, str] = {}
    for row in reform_rows:
        gov_uri = str(row.governor)
        aliases = {str(row.govLabelSi or "").lower(), str(row.govLabelEn or "").lower()}
        aliases = {a for a in aliases if a}
        if any(alias in answer_text for alias in aliases):
            mentioned_governors[gov_uri] = known_governor_labels[gov_uri]

    for row in reform_rows:
        reform_uri = str(row.reform)
        reform_si = str(row.reformLabelSi or "")
        reform_en = str(row.reformLabelEn or "")
        reform_aliases = [reform_si.lower(), reform_en.lower(), "colebrooke", "cameron", "1833"]
        if not any(alias and alias in answer_text for alias in reform_aliases):
            continue

        expected_gov_uri = str(row.governor)
        expected_gov = str(row.govLabelSi or row.govLabelEn or expected_gov_uri)
        checked_claims.append(
            {
                "claim_type": "reform_governor",
                "entity": reform_si or reform_en or reform_uri,
                "expected_relation": expected_gov,
                "ontology_rule": f"{expected_gov_uri} ex:implemented {reform_uri}",
            }
        )

        wrong_govs = [g_uri for g_uri in mentioned_governors.keys() if g_uri != expected_gov_uri]
        if wrong_govs:
            mismatches.append(
                {
                    "claim_type": "reform_governor",
                    "entity": reform_si or reform_en or reform_uri,
                    "expected": expected_gov,
                    "found": ", ".join(mentioned_governors[g_uri] for g_uri in wrong_govs),
                    "ontology_rule": f"{expected_gov_uri} ex:implemented {reform_uri}",
                }
            )

        # Derived period rule for colonial power alignment of this reform.
        british_uri = "http://example.org/colonial#British"
        if british_uri in power_aliases:
            wrong_powers = [p for p in mentioned_powers.keys() if p != british_uri]
            if wrong_powers:
                mismatches.append(
                    {
                        "claim_type": "reform_colonial_power",
                        "entity": reform_si or reform_en or reform_uri,
                        "expected": power_aliases[british_uri]["label_si"] or power_aliases[british_uri]["label_en"],
                        "found": ", ".join(mentioned_powers[p] for p in wrong_powers),
                        "ontology_rule": (
                            "ex:EdwardBarnes ex:implemented ex:ColebrookeCameron1833 ; "
                            "ex:Dutch ex:succeeded_by ex:British"
                        ),
                    }
                )

    historical_penalty = min(4, len(mismatches))
    return {
        "question_id": question_obj["id"],
        "checked_claims": checked_claims,
        "mismatches": mismatches,
        "historical_inaccuracy_penalty": historical_penalty,
        "requires_deduction": historical_penalty > 0,
    }


def retrieval_agent(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    keywords = state["keywords"]
    query = f"{question['question_si']} {' '.join(keywords)} {state['student_answer_si']}"
    rag_hits = retrieve(query=query, out_dir=Path("vector_store"), k=5)
    onto_facts = ontology_context_for_question(question)
    semantic_report = semantic_verify_with_ontology(question, state["student_answer_si"])
    return {
        **state,
        "rag_hits": rag_hits,
        "ontology_facts": onto_facts,
        "semantic_verification": semantic_report,
    }


def ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    #r = requests.post(url, json=payload, timeout=180)
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

    prompt = f"""
You are a strict Sinhala history grader.
Question (Sinhala): {question['question_si']}
Marking guide JSON: {json.dumps(guide, ensure_ascii=False)}
Student answer (Sinhala): {answer}

Retrieved textbook evidence:
{json.dumps(rag_hits, ensure_ascii=False)}

Ontology facts:
{json.dumps(onto_facts, ensure_ascii=False)}

Semantic verification report (from rdflib helper query):
{json.dumps(semantic_report, ensure_ascii=False)}

Reviewer correction feedback (if present):
{reviewer_feedback or "N/A"}

Task:
1) Perform semantic verification FIRST using the provided rdflib helper report before assigning marks.
2) If semantic_verification.requires_deduction is true, deduct exactly semantic_verification.historical_inaccuracy_penalty marks for historical inaccuracy.
3) The deduction must be applied to the historical-accuracy criterion, and explanation_si must cite the exact ontology rule string using [ONTO rule=...].
4) Score each criterion independently using ONLY the marking guide maxima.
5) Give Sinhala explanation per criterion with evidence reference and at least one citation in format [RAG rank=<n> source=<path> chunk=<id>].
6) If no mismatch exists, explicitly state there is no ontology contradiction.
7) Return valid JSON only with structure:
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

    return "to_critic"


def critic_agent(state: ScoreState) -> ScoreState:
    guide = state["marking_guide"]
    result = state["grading_result"]
    rag_hits = state.get("rag_hits", [])
    expected_max_total = int(guide.get("total_marks", 20))
    criteria = result.get("criteria_scores", [])
    issues: List[str] = []

    guide_by_id = {c["criterion_id"]: int(c["marks"]) for c in guide["criteria"]}
    if len(criteria) != len(guide_by_id):
        issues.append("criterion_count_mismatch")

    awarded_total = 0
    max_total = 0
    weak_criteria: List[str] = []
    citation_regex = re.compile(r"\[RAG\s+rank=(\d+)\s+source=(.+?)\s+chunk=(\d+)\]")
    valid_citation_keys = {(h["source"], int(h["chunk_id"])) for h in rag_hits}

    for item in criteria:
        cid = item.get("criterion_id", "")
        awarded = int(item.get("awarded_marks", 0))
        cmax = int(item.get("max_marks", 0))
        explanation = str(item.get("explanation_si", ""))
        awarded_total += awarded
        max_total += cmax

        if cid in guide_by_id and cmax != guide_by_id[cid]:
            issues.append(f"max_marks_mismatch:{cid}")

        matches = citation_regex.findall(explanation)
        has_valid = False
        for _, source, chunk in matches:
            if (source.strip(), int(chunk)) in valid_citation_keys:
                has_valid = True
                break
        if not has_valid:
            weak_criteria.append(cid)

        if not re.search(r"[\u0D80-\u0DFF]", explanation):
            issues.append(f"non_sinhala_explanation:{cid}")

    if awarded_total != int(result.get("total_score", -1)):
        issues.append("awarded_total_mismatch")
    if max_total != expected_max_total:
        issues.append("max_total_not_20")

    grounding_ratio = (len(criteria) - len(weak_criteria)) / max(1, len(criteria))
    if grounding_ratio < 0.6:
        issues.append("weak_grounding")

    reviewer_pass = len(issues) == 0
    feedback = (
        "Revise grading output: "
        + "; ".join(issues)
        + ". Ensure mark arithmetic is exact and every criterion explanation has a valid [RAG ...] citation."
    )
    report = {
        "pass": reviewer_pass,
        "issues": issues,
        "grounding_ratio": round(grounding_ratio, 3),
        "weak_criteria": weak_criteria,
        "awarded_total": awarded_total,
        "max_total": max_total,
    }
    return {**state, "critic_report": report, "critic_feedback": "" if reviewer_pass else feedback}


def critic_quality_gate(state: ScoreState) -> str:
    retries = state.get("critic_retries", 0)
    report = state.get("critic_report", {})
    if report.get("pass"):
        return "to_justification"
    if retries < 1:
        state["critic_retries"] = retries + 1
        return "retry_grading"
    raise ValueError(f"Critic rejected grading output: {', '.join(report.get('issues', []))}")


def justification_agent(state: ScoreState) -> ScoreState:
    grading = state["grading_result"]
    rag_hits = state.get("rag_hits", [])
    onto_facts = state.get("ontology_facts", [])
    semantic_report = state.get("semantic_verification", {})
    critic_report = state.get("critic_report", {})

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
        "semantic_verification": semantic_report,
        "critic_report": critic_report,
    }
    return {**state, "grading_result": grading}


def confidence_agent(state: ScoreState) -> ScoreState:
    rag_hits = state.get("rag_hits", [])
    semantic_report = state.get("semantic_verification", {})
    critic_report = state.get("critic_report", {})
    guide = state["marking_guide"]

    top_scores = [float(hit["score"]) for hit in rag_hits[:3]]
    avg_similarity = sum(top_scores) / max(1, len(top_scores))
    normalized_similarity = max(0.0, min(1.0, (avg_similarity + 1.0) / 2.0))
    retrieval_coverage = max(0.0, min(1.0, len(rag_hits) / 5.0))
    grounding_ratio = float(critic_report.get("grounding_ratio", 0.0))
    mismatch_count = len(semantic_report.get("mismatches", []))
    semantic_consistency = max(0.0, 1.0 - (min(4, mismatch_count) / 4.0))

    confidence = 100.0 * (
        0.35 * normalized_similarity
        + 0.25 * retrieval_coverage
        + 0.25 * grounding_ratio
        + 0.15 * semantic_consistency
    )
    confidence = round(max(0.0, min(100.0, confidence)), 1)

    weak_ids = set(critic_report.get("weak_criteria", []))
    low_sections: List[Dict[str, str]] = []
    for criterion in guide["criteria"]:
        cid = criterion["criterion_id"]
        reason_parts: List[str] = []
        if cid in weak_ids:
            reason_parts.append("FAISS context citation is weak or missing")
        if semantic_report.get("requires_deduction") and (
            "ඉතිහාසික" in criterion["description_si"] or "නිවැරදි" in criterion["description_si"]
        ):
            reason_parts.append("Ontology mismatch triggered historical inaccuracy deduction")
        if retrieval_coverage < 0.6:
            reason_parts.append("Retrieved context is sparse")
        if reason_parts:
            low_sections.append(
                {
                    "criterion_id": cid,
                    "description_si": criterion["description_si"],
                    "reason": " | ".join(reason_parts),
                }
            )

    confidence_payload = {
        "score": confidence,
        "level": "High" if confidence >= 75 else "Medium" if confidence >= 60 else "Low",
        "low_confidence_sections": low_sections,
    }
    return {**state, "grading_result": {**state["grading_result"], "confidence": confidence_payload}}


def output_formatter(state: ScoreState) -> ScoreState:
    question = state["question_obj"]
    out = {
        "question_id": question["id"],
        "question_si": question["question_si"],
        "score_out_of_20": state["grading_result"]["total_score"],
        "criteria_breakdown": state["grading_result"]["criteria_scores"],
        "final_feedback_si": state["grading_result"]["final_feedback_si"],
        "evidence": state["grading_result"]["evidence_summary"],
        "confidence": state["grading_result"]["confidence"],
    }
    return {**state, "final_output": out}


def build_graph():
    graph = StateGraph(ScoreState)
    graph.add_node("orchestrator", load_question_from_config)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("grading_agent", grading_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node("justification_agent", justification_agent)
    graph.add_node("confidence_agent", confidence_agent)
    graph.add_node("output_formatter", output_formatter)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "retrieval_agent")
    graph.add_edge("retrieval_agent", "grading_agent")
    graph.add_conditional_edges(
        "grading_agent",
        grade_quality_gate,
        {
            "retry_grading": "grading_agent",
            "to_critic": "critic_agent",
        },
    )
    graph.add_conditional_edges(
        "critic_agent",
        critic_quality_gate,
        {
            "retry_grading": "grading_agent",
            "to_justification": "justification_agent",
        },
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
