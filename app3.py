"""
Document Intelligence: Advanced Contract Analysis and Chat Application

Description:
This application uses the Gemini API for intelligent contract analysis and interaction. It supports PDF, DOCX, and TXT files, enabling:
1. **Semantic Clause Comparison**: Pairs clauses between contracts (e.g., MSA and Vendor) using semantic similarity, even for non-exact matches, highlighting differences, risks, and suggestions.
2. **Regulatory Compliance Check**: Identifies missing or non-compliant rules/regulations based on standard contract frameworks.
3. **Contract Compliance & Quality Assessment**: Evaluates the uploaded contract against the MSA for clause coverage and judges its quality (clarity, completeness, enforceability).
4. **Executive Summary**: Provides a high-level overview of findings, including key differences, risks, compliance issues, and contract quality, for stakeholder review.
5. **Contextual Q&A**: Answers questions about contracts using semantic search for relevant clauses.
6. **Sample Contract Loading**: Provides sample contracts for testing.
The tool uses embeddings for semantic matching, cosine similarity for relevance, and Streamlit for a professional UI, with robust error handling and progress tracking.
"""

import streamlit as st
import os
import docx
import PyPDF2
import re
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import google.generativeai as genai
import time

# Load API Key
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {str(e)}")
    st.stop()

# Utility functions
def extract_text_docx(file) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_pdf(file) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_txt(file) -> str:
    """Extract text from a TXT file."""
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, filtering out short or empty ones."""
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.split()) > 10]
    return paragraphs

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a given text."""
    try:
        response = model.embed_content(content=text, task_type="RETRIEVAL_DOCUMENT")
        return response["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return []

def cosine_similarity(vecA: List[float], vecB: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        vecA, vecB = np.array(vecA), np.array(vecB)
        return np.dot(vecA, vecB) / (norm(vecA) * norm(vecB))
    except:
        return 0.0

def analyze_clause(msa_clause: str, vendor_clause: str, similarity: float) -> Dict:
    """Analyze differences, risks, suggestions, and compliance for two clauses."""
    prompt = f"""
You are an expert contract analyst. Compare these clauses semantically:

**MSA Clause:**
{msa_clause}

**Vendor Clause:**
{vendor_clause}

**Similarity Score:**
{similarity:.2f}

Provide a structured response with:
1. **Key Differences**: Highlight semantic and textual variations.
2. **Risks**: Identify legal, financial, or operational risks in the vendor clause.
3. **Suggestions**: Recommend improvements or negotiation points.
4. **Regulatory Compliance**: Check for missing or non-compliant rules/regulations (e.g., GDPR, UCC, industry standards).
Keep it concise and professional.
"""
    try:
        response = model.generate_content(prompt)
        return {
            "analysis": response.text,
            "compliance_issues": "Non-compliant" in response.text.lower() or "missing" in response.text.lower(),
            "risks": "Risks" in response.text
        }
    except Exception as e:
        return {"analysis": f"Analysis error: {str(e)}", "compliance_issues": False, "risks": False}

def evaluate_contract_compliance(msa_paragraphs: List[str], vendor_paragraphs: List[str]) -> Dict:
    """Evaluate vendor contract compliance and quality against MSA."""
    prompt = f"""
You are a contract quality assessor. Evaluate the vendor contract's compliance with the MSA and its overall quality.

**MSA Clauses:**
{' '.join(msa_paragraphs[:5]) + '...'}

**Vendor Clauses:**
{' '.join(vendor_paragraphs[:5]) + '...'}

Provide:
1. **Compliance Score**: Percentage of MSA clauses covered (0-100).
2. **Missing Clauses**: List key MSA clauses not found in the vendor contract.
3. **Quality Assessment**:
   - Clarity: Is the language clear and unambiguous?
   - Completeness: Are critical terms (e.g., payment, termination) included?
   - Enforceability: Is the contract legally sound?
4. **Summary**: Brief overall judgment.
"""
    try:
        response = model.generate_content(prompt)
        score = float(re.search(r"Compliance Score: (\d+)%", response.text).group(1)) if re.search(r"Compliance Score: (\d+)%", response.text) else 0
        return {
            "text": response.text,
            "score": score,
            "missing_clauses": re.findall(r"- (.*?)(?:\n|$)", response.text) if "Missing Clauses" in response.text else []
        }
    except Exception as e:
        return {"text": f"Evaluation error: {str(e)}", "score": 0, "missing_clauses": []}

def generate_executive_summary(
    results: List[Tuple], unmatched_msa: List[Tuple], evaluation: Dict
) -> str:
    """Generate an executive summary of the contract analysis."""
    high_risk_count = sum(1 for r in results if r[4]["risks"])
    compliance_issues = sum(1 for r in results if r[4]["compliance_issues"])
    unmatched_count = len(unmatched_msa)
    compliance_score = evaluation.get("score", 0)
    missing_clauses = evaluation.get("missing_clauses", [])

    prompt = f"""
You are a contract analysis expert preparing an executive summary for stakeholders.

**Analysis Overview:**
- **Clause Matches Analyzed**: {len(results)} clauses compared semantically.
- **High-Risk Clauses**: {high_risk_count} clauses with significant risks.
- **Compliance Issues**: {compliance_issues} clauses with potential regulatory concerns.
- **Unmatched MSA Clauses**: {unmatched_count} MSA clauses without a vendor match.
- **Compliance Score**: {compliance_score}% of MSA clauses covered by the vendor contract.
- **Missing Clauses**: {', '.join(missing_clauses) if missing_clauses else 'None identified.'}

**Task:**
Provide a concise executive summary (150-200 words) covering:
1. **Key Findings**: Summary of clause alignment and critical differences.
2. **Risks and Compliance**: Highlight major risks and regulatory gaps.
3. **Contract Quality**: Overview of compliance score and quality assessment.
4. **Recommendations**: High-level suggestions for next steps.
Use a professional tone suitable for senior management.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def chat_with_contract(query: str, paragraphs: List[str]) -> str:
    """Answer a query using relevant contract paragraphs."""
    try:
        query_emb = get_embedding(query)
        if not query_emb:
            return "Failed to process query."
        para_scores = [(p, cosine_similarity(get_embedding(p), query_emb)) for p in paragraphs]
        best = sorted(para_scores, key=lambda x: x[1], reverse=True)[:3]
        context = "\n\n".join([b[0] for b in best])
        prompt = f"""
**Context:**
{context}

**Question:**
{query}

Provide a clear, concise answer based on the context. If the context is insufficient, state so and suggest next steps.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Chat error: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Document Intelligence", layout="wide", page_icon="📑")
st.title("📑 Document Intelligence")
st.markdown("Analyze contracts semantically, check compliance, assess quality, and generate executive summaries with AI-powered insights.")

# Initialize session state
if "msa_text" not in st.session_state:
    st.session_state["msa_text"] = ""
if "vendor_text" not in st.session_state:
    st.session_state["vendor_text"] = ""
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = []
if "unmatched_msa" not in st.session_state:
    st.session_state["unmatched_msa"] = []
if "evaluation" not in st.session_state:
    st.session_state["evaluation"] = {}

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Clause Comparison", "✅ Compliance & Quality", "💬 Contract Q&A", "📚 Load Samples"])

with tab4:
    st.subheader("Load Sample Contracts")
    if st.button("Load Example MSA & Vendor Contracts"):
        with st.spinner("Loading sample contracts..."):
            try:
                with open("examples/sample_msa.pdf", "rb") as f:
                    st.session_state["msa_text"] = extract_text_pdf(f)
                with open("examples/sample_vendor.pdf", "rb") as f:
                    st.session_state["vendor_text"] = extract_text_pdf(f)
                st.success("Sample contracts loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load samples: {str(e)}")

with tab1:
    st.subheader("Semantic Clause Comparison")
    col1, col2 = st.columns(2)
    with col1:
        msa_file = st.file_uploader("Upload MSA Contract", type=["pdf", "docx", "txt"], key="msa")
    with col2:
        vendor_file = st.file_uploader("Upload Vendor Contract", type=["pdf", "docx", "txt"], key="vendor")

    if msa_file:
        with st.spinner("Processing MSA contract..."):
            if msa_file.type == "application/pdf":
                st.session_state["msa_text"] = extract_text_pdf(msa_file)
            elif msa_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                st.session_state["msa_text"] = extract_text_docx(msa_file)
            else:
                st.session_state["msa_text"] = extract_text_txt(msa_file)
            if st.session_state["msa_text"]:
                st.success("MSA contract processed.")

    if vendor board:
        with st.spinner("Processing Vendor contract..."):
            if vendor_file.type == "application/pdf":
                st.session_state["vendor_text"] = extract_text_pdf(vendor_file)
            elif vendor_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                st.session_state["vendor_text"] = extract_text_docx(vendor_file)
            else:
                st.session_state["vendor_text"] = extract_text_txt(vendor_file)
            if st.session_state["vendor_text"]:
                st.success("Vendor contract processed.")

    if st.session_state["msa_text"] and st.session_state["vendor_text"]:
        msa_paragraphs = split_into_paragraphs(st.session_state["msa_text"])
        vendor_paragraphs = split_into_paragraphs(st.session_state["vendor_text"])

        if st.button("Compare Clauses Semantically"):
            with st.spinner("Analyzing clauses with semantic matching..."):
                progress = st.progress(0)
                results = []
                unmatched_msa = []
                total = len(msa_paragraphs)
                for i, msa_para in enumerate(msa_paragraphs):
                    try:
                        msa_emb = get_embedding(msa_para)
                        if not msa_emb:
                            unmatched_msa.append((i+1, msa_para))
                            continue
                        similarities = [
                            (j, vp, cosine_similarity(msa_emb, get_embedding(vp)))
                            for j, vp in enumerate(vendor_paragraphs)
                        ]
                        similarities = [(j, vp, score) for j, vp, score in similarities if score >= 0.5]
                        if similarities:
                            for j, ven_para, score in sorted(similarities, key=lambda x: x[2], reverse=True):
                                analysis = analyze_clause(msa_para, ven_para, score)
                                results.append((i+1, msa_para, ven_para, score, analysis))
                        else:
                            unmatched_msa.append((i+1, msa_para))
                    except:
                        unmatched_msa.append((i+1, msa_para))
                    progress.progress((i + 1) / total)

                st.session_state["analysis_results"] = results
                st.session_state["unmatched_msa"] = unmatched_msa

                if results:
                    st.subheader("Semantic Clause Matches")
                    for idx, msa_p, ven_p, score, analysis in sorted(results, key=lambda x: x[3], reverse=True):
                        with st.expander(f"Clause Match {idx} (Similarity: {score:.2f})"):
                            st.markdown("**MSA Clause:**")
                            st.write(msa_p)
                            st.markdown("**Vendor Clause:**")
                            st.write(ven_p)
                            st.markdown("**Analysis:**")
                            st.markdown(analysis["analysis"])
                            if analysis["compliance_issues"]:
                                st.warning("Potential compliance issues detected.")
                            if analysis["risks"]:
                                st.error("High-risk clause detected.")
                else:
                    st.warning("No significant clause matches found (similarity < 0.5).")

                if unmatched_msa:
                    st.subheader("Unmatched MSA Clauses")
                    for idx, msa_p in unmatched_msa:
                        with st.expander(f"Unmatched Clause {idx}"):
                            st.markdown("**MSA Clause:**")
                            st.write(msa_p)
                            st.info("No semantically similar clause found in the vendor contract.")

                progress.empty()

with tab2:
    st.subheader("Compliance, Quality & Executive Summary")
    if st.session_state["msa_text"] and st.session_state["vendor_text"]:
        if st.button("Evaluate Contract & Generate Summary"):
            with st.spinner("Evaluating compliance, quality, and generating summary..."):
                msa_paragraphs = split_into_paragraphs(st.session_state["msa_text"])
                vendor_paragraphs = split_into_paragraphs(st.session_state["vendor_text"])
                evaluation = evaluate_contract_compliance(msa_paragraphs, vendor_paragraphs)
                st.session_state["evaluation"] = evaluation

                st.markdown("**Evaluation Results:**")
                st.markdown(evaluation["text"])
                score = evaluation["score"]
                st.metric("Compliance Score", f"{score}%")
                if score < 70:
                    st.error("Low compliance score. Review missing clauses and risks.")
                elif score < 90:
                    st.warning("Moderate compliance. Consider addressing gaps.")
                else:
                    st.success("High compliance with MSA.")

                if st.session_state["analysis_results"] or st.session_state["unmatched_msa"]:
                    st.subheader("Executive Summary")
                    summary = generate_executive_summary(
                        st.session_state["analysis_results"],
                        st.session_state["unmatched_msa"],
                        st.session_state["evaluation"]
                    )
                    st.markdown(summary)
                    st.download_button(
                        label="Download Executive Summary",
                        data=summary,
                        file_name="executive_summary.txt",
                        mime="text/plain"
                    )
    else:
        st.warning("Please upload both MSA and Vendor contracts to evaluate.")

with tab3:
    st.subheader("Ask Questions About Contracts")
    if st.session_state["msa_text"] or st.session_state["vendor_text"]:
        paragraphs = split_into_paragraphs(st.session_state["msa_text"]) + split_into_paragraphs(st.session_state["vendor_text"])
        query = st.text_input("Enter your question about the contracts", placeholder="e.g., What are the termination clauses?")
        if query:
            with st.spinner("Generating response..."):
                response = chat_with_contract(query, paragraphs)
                st.markdown("**Answer:**")
                st.markdown(response)
    else:
        st.warning("Please upload or load at least one contract to enable Q&A.")
