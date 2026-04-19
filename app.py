import json
from pathlib import Path
import pandas as pd
import streamlit as st
from scorer_workflow import run_scoring
import warnings
import logging

# Suppress logs and warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# --- Page Configuration & Custom Styling ---
st.set_page_config(page_title="ඉතිහාස ගුරු - Ithihaasa Guru", layout="wide")

# Inject Custom CSS for a modern, history-friendly, and Sri Lankan feel
st.markdown("""
    <style>
    /* 1. App Background - Warm Cream/Parchment shade */
    .stApp {
        background-color: #fdfaf0;
    }
    
    /* 2. Custom Question Box - Light Orange/Tan (Replacing the Blue Info box) */
    .question-container {
        background-color: #fff3e0;
        border-left: 8px solid #ff9800;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        color: #5d4037;
        font-size: 1.2rem;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    /* 3. Title Styling - Sri Lankan Maroon */
    h1 {
        color: #800000; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        font-weight: 800;
        padding-bottom: 10px;
    }
    
    /* 4. WhatsApp Green Button */
    .stButton>button {
        background-color: #25D366 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: bold !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #128C7E !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Result Card Styling */
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-top: 5px solid #25D366;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
config = json.loads(Path("history_config.json").read_text(encoding="utf-8"))
question_map = {q["id"]: q for q in config["questions"]}
question_ids = list(question_map.keys())

# --- Sidebar / Header ---
st.title("ඉතිහාස ගුරු (Ithihaasa Guru)")
st.markdown("<p style='text-align: center; color: #555;'>ශ්‍රී ලංකාවේ යටත්විජිත ඉතිහාසය පිළිබඳ ඉගෙනුම් සහ ඇගයීම් පද්ධතිය</p>", unsafe_allow_html=True)

# Question Selection
selected_qid = st.selectbox("ප්‍රශ්නය තෝරන්න", options=question_ids, format_func=lambda qid: question_map[qid]["question_si"])
selected_q = question_map[selected_qid]


# Displaying the Question in the new Light Orange Box
st.markdown(f"""
    <div class="question-container">
        <strong>ප්‍රශ්නය:</strong><br>
        {selected_q['question_si']}
    </div>
    """, unsafe_allow_html=True)

# --- Student Input Section ---
student_answer = st.text_area("ඔබේ පිළිතුර මෙහි ඇතුළත් කරන්න (Sinhala)", height=250, placeholder="පෘතුගීසීන් ලංකාවට පැමිණි ආකාරය සහ...")

# Action Button
if st.button("ලකුණු පරීක්ෂා කරන්න"):
    if not student_answer.strip():
        st.error("කරුණාකර පිළිතුරක් ඇතුළත් කරන්න.")
    else:
        with st.spinner("ඔබේ පිළිතුර ඇගයීමට ලක්වෙමින් පවතී..."):
            try:
                result = run_scoring(question_id=selected_qid, student_answer_si=student_answer.strip())
            except Exception as e:
                st.exception(e)
            else:
                # --- RESULTS DISPLAY ---
                st.success("ඇගයීම සාර්ථකයි!")
                
                # Top Metrics
                confidence = result.get("confidence", {})
                confidence_score = confidence.get("score", 0.0)
                
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    st.metric("ලැබුණු ලකුණු", f"{result['score_out_of_20']} / 20")
                with c2:
                    st.markdown(f"<div class='result-card'><strong>පද්ධති ප්‍රතිචාරය:</strong><br>{result['final_feedback_si']}</div>", unsafe_allow_html=True)
                with c3:
                    st.metric("විශ්වාස මට්ටම", f"{confidence_score:.1f}%")

                # --- NEW LOGIC: Show Marking Scheme ONLY AFTER answering ---
                with st.expander("ලකුණු ලබාදීමේ පදනම (Marking Guide)", expanded=False):
                    guide_rows = [
                        {"Criterion": c["criterion_id"], "විස්තරය": c["description_si"], "Marks": c["marks"]}
                        for c in selected_q["marking_guide"]["criteria"]
                    ]
                    st.table(pd.DataFrame(guide_rows))

                # Detailed Breakdown
                st.subheader("විස්තරාත්මක ලකුණු බෙදාහැරීම")
                breakdown_rows = [
                    {
                        "නිර්ණායකය": item["criterion_id"],
                        "ලකුණු": item["awarded_marks"],
                        "උපරිම": item["max_marks"],
                        "විස්තරය": item["explanation_si"],
                    }
                    for item in result["criteria_breakdown"]
                ]
                st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

                # Low Confidence Warnings
                if confidence_score < 60:
                    st.warning("මෙම පිළිතුරට අදාළව නිශ්චිත සාක්ෂි සොයාගැනීම අපහසු බැවින් විශ්වාස මට්ටම අඩු විය හැකිය.")
                    weak_rows = confidence.get("low_confidence_sections", [])
                    if weak_rows:
                        st.dataframe(pd.DataFrame(weak_rows), use_container_width=True, hide_index=True)

                # Evidence Section
                with st.expander("ඓතිහාසික සාක්ෂි (Evidence-based Explanation)"):
                    st.write("**පොත්පත් සහ ලේඛන මූලාශ්‍ර (RAG Top Hits)**")
                    for i, hit in enumerate(result["evidence"]["rag_top_hits"], start=1):
                        source = hit.get("source", "Unknown Source")
                        snippet = hit.get("snippet", hit.get("text", "No content available"))
                        st.markdown(f"**{i}. මූලාශ්‍රය: {source}**\n> {snippet}")
                    
                    st.write("---")
                    st.write("**ඓතිහාසික කරුණු (Ontology Facts)**")
                    for fact in result["evidence"]["ontology_facts"]:
                        st.markdown(f"- {fact}")