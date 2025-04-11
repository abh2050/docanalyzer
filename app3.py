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

def get_embedding(text: str, max_length: int = 2500) -> List[float]:
    """
    Generate embedding for a given text.
    Truncates text if it exceeds max_length to avoid API payload limits.
    """
    try:
        # Truncate text if it's too long to avoid payload size errors
        if len(text) > max_length:
            text = text[:max_length]
            
        # Using embedding-001 model for embeddings
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="semantic_similarity"
        )
        return result["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return []

def embed_long_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[float]:
    """
    Generate embedding for long texts by splitting into chunks, 
    embedding each chunk, and averaging the results.
    """
    if len(text) <= chunk_size:
        return get_embedding(text)
    
    # Split text into overlapping chunks
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only include chunks with meaningful content
            chunks.append(chunk)
    
    if not chunks:
        return []
    
    # Get embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb:
            embeddings.append(np.array(emb))
    
    if not embeddings:
        return []
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding.tolist()

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

def evaluate_contract_and_generate_results(msa_paragraphs: List[str], vendor_paragraphs: List[str]):
    """Evaluate contract and generate analysis results."""
    evaluation = evaluate_contract_compliance(msa_paragraphs, vendor_paragraphs)
    
    # Create placeholder analysis results for executive summary
    st.session_state["analysis_results"] = []
    st.session_state["unmatched_msa"] = []
    st.session_state["evaluation"] = evaluation
    
    return evaluation

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
        # Use shorter paragraphs or truncate very long ones
        filtered_paragraphs = []
        for p in paragraphs:
            if len(p) > 7500:  # Truncate extremely long paragraphs (increased from 5000)
                p = p[:7500] + "..."
            filtered_paragraphs.append(p)
        
        # Get query embedding
        query_emb = get_embedding(query)
        if not query_emb:
            return "Failed to process query."
        
        # Calculate similarity for each paragraph
        para_scores = []
        for p in filtered_paragraphs:
            try:
                # Use the embed_long_text function for potentially long paragraphs
                p_emb = embed_long_text(p) if len(p) > 2000 else get_embedding(p)
                if p_emb:
                    score = cosine_similarity(p_emb, query_emb)
                    para_scores.append((p, score))
            except Exception as e:
                st.error(f"Error processing paragraph: {str(e)}")
        
        if not para_scores:
            return "Could not analyze the contracts effectively. Please try a different query."
        
        # Sort by similarity score and get top 5 (increased from 3)
        best = sorted(para_scores, key=lambda x: x[1], reverse=True)[:5]
        
        # Limit context size but increased for up to 10 pages
        context_parts = []
        total_length = 0
        max_context_length = 10000  # Increased from 6000 (about 10 pages of content)
        
        for b in best:
            if total_length + len(b[0]) <= max_context_length:
                context_parts.append(b[0])
                total_length += len(b[0])
            else:
                # Add a truncated version if the full paragraph would exceed limit
                space_left = max_context_length - total_length
                if space_left > 500:  # Only add if we can include something substantial
                    context_parts.append(b[0][:space_left] + "...")
                break
        
        context = "\n\n".join(context_parts)
        
        # Craft prompt carefully to stay within limits
        prompt = f"""
**Context:**
{context}

**Question:**
{query}

Provide a clear, concise answer based on the context. If the context is insufficient, state so and suggest next steps.
"""
        response = model.generate_content(prompt)
        
        # Add disclaimer to the response
        disclaimer = "\n\n*Disclaimer: This AI-generated response is based on contract analysis and should not be considered legal advice. Please consult with qualified legal counsel for professional guidance on contract matters.*"
        
        return response.text + disclaimer
    except Exception as e:
        return f"Chat error: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Document Intelligence", layout="wide", page_icon="📑")
st.title("📑 Document Intelligence")
st.markdown("Analyze contracts, check compliance, assess quality, and generate executive summaries with AI-powered insights.")

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

# Modified tabs without the Load Samples tab
tab1, tab2 = st.tabs(["✅ Compliance, Quality & Executive Summary", "💬 Contract Q&A"])

with tab1:
    st.subheader("Compliance, Quality & Executive Summary")
    
    # Add file uploaders for contracts and sample loading option
    col1, col2 = st.columns(2)
    
    with col1:
        msa_file = st.file_uploader("Upload MSA Contract", type=["pdf", "docx", "txt"], key="msa")
    with col2:
        vendor_file = st.file_uploader("Upload Vendor Contract", type=["pdf", "docx", "txt"], key="vendor")
    
    # Add the Load Sample button
    if st.button("Load Example MSA & Vendor Contracts"):
        with st.spinner("Loading sample contracts..."):
            try:
                # Use relative paths for GitHub/Streamlit Cloud compatibility
                examples_dir = "examples"
                msa_path = os.path.join(examples_dir, "MASTER SERVICES AGREEMENT.docx")
                vendor_path = os.path.join(examples_dir, "VENDOR CONTRACT AGREEMENT.docx")
                
                # Create examples directory if it doesn't exist
                if not os.path.exists(examples_dir):
                    os.makedirs(examples_dir)
                    st.info(f"Created directory: {examples_dir}")
                
                # Check if files exist
                if os.path.exists(msa_path):
                    st.session_state["msa_text"] = extract_text_docx(msa_path)
                    if st.session_state["msa_text"]:
                        st.success("MSA contract loaded successfully!")
                else:
                    st.error(f"MSA file not found at: {msa_path}")
                
                if os.path.exists(vendor_path):
                    st.session_state["vendor_text"] = extract_text_docx(vendor_path)
                    if st.session_state["vendor_text"]:
                        st.success("Vendor contract loaded successfully!")
                else:
                    st.error(f"Vendor file not found at: {vendor_path}")
                    
            except Exception as e:
                st.error(f"Failed to load samples: {str(e)}")

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

    if vendor_file:
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

        if st.button("Evaluate Contract & Generate Summary"):
            with st.spinner("Evaluating compliance, quality, and generating summary..."):
                evaluation = evaluate_contract_and_generate_results(msa_paragraphs, vendor_paragraphs)

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
                
                st.subheader("Executive Summary")
                summary = generate_executive_summary(
                    st.session_state["analysis_results"],
                    st.session_state["unmatched_msa"],
                    evaluation
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

with tab2:
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
