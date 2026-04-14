import json
from pathlib import Path

import pandas as pd
import streamlit as st

from scorer_workflow import run_scoring


st.set_page_config(page_title="Sinhala Colonial History Scorer", layout="wide")
st.title("ශ්‍රී ලංකාවේ යටත්විජිත ඉතිහාසය - විවෘත පිළිතුරු ලකුණුකරණය")
st.caption("Offline scoring with LangGraph + FAISS + Ontology + Ollama (gemma3:12b)")

config = json.loads(Path("history_config.json").read_text(encoding="utf-8"))
question_map = {q["id"]: q for q in config["questions"]}
question_ids = list(question_map.keys())

selected_qid = st.selectbox("ප්‍රශ්නය තෝරන්න", options=question_ids, format_func=lambda qid: question_map[qid]["question_si"])
selected_q = question_map[selected_qid]

st.subheader("තෝරාගත් ප්‍රශ්නය")
st.write(selected_q["question_si"])

st.subheader("ලකුණු මාර්ගෝපදේශය (20)")
guide_rows = [
    {"Criterion ID": c["criterion_id"], "විස්තරය": c["description_si"], "Marks": c["marks"]}
    for c in selected_q["marking_guide"]["criteria"]
]
st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)

student_answer = st.text_area("සිසුවාගේ පිළිතුර (Sinhala)", height=220)

if st.button("ලකුණු කරන්න", type="primary"):
    if not student_answer.strip():
        st.error("කරුණාකර පිළිතුරක් ඇතුළත් කරන්න.")
    else:
        with st.spinner("ලකුණුකරණය සිදු වෙමින් පවතී..."):
            try:
                result = run_scoring(question_id=selected_qid, student_answer_si=student_answer.strip())
            except Exception as e:
                st.exception(e)
            else:
                st.success("ලකුණුකරණය සාර්ථකයි.")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("මුළු ලකුණු", f"{result['score_out_of_20']} / 20")
                with c2:
                    st.write("**අවසාන ප්‍රතිචාරය**")
                    st.write(result["final_feedback_si"])

                st.subheader("විස්තරාත්මක ලකුණු බෙදාහැරීම")
                breakdown_rows = []
                for item in result["criteria_breakdown"]:
                    breakdown_rows.append(
                        {
                            "Criterion": item["criterion_id"],
                            "ලැබුණු ලකුණු": item["awarded_marks"],
                            "උපරිම": item["max_marks"],
                            "විස්තරය": item["explanation_si"],
                        }
                    )
                st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

                st.subheader("සාක්ෂි-මත පැහැදිලි කිරීම")
                st.write("**RAG Top Hits**")
                for i, hit in enumerate(result["evidence"]["rag_top_hits"], start=1):
                    st.markdown(
                        f"{i}. **මූලාශ්‍රය:** `{hit['source']}` (chunk: {hit['chunk_id']}, sim: {hit['score']:.3f})\n\n"
                        f"> {hit['snippet']}"
                    )
                st.write("**Ontology Facts**")
                for fact in result["evidence"]["ontology_facts"]:
                    st.markdown(f"- {fact}")
