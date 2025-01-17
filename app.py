import streamlit as st
from io import BytesIO
import docx
import PyPDF2
import openai
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

# ------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ------------------------------------------------------------

# Load the environment variables from .env file
load_dotenv('.env')

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure NLTK data is downloaded
nltk.download('punkt')

# ------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------

# Extract text from DOCX
def extract_text_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract text from PDF
def extract_text_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Split text into clauses based on common legal numbering
def split_into_clauses(text):
    # Regex to capture clauses like "\n1. ", "\n2.1 ", etc.
    clause_pattern = re.compile(r'(\n\d+(\.\d+)*\s)')
    parts = clause_pattern.split(text)
    
    combined_clauses = []
    i = 1
    while i < len(parts):
        number = parts[i].strip()
        clause_text = ""
        if (i + 2) < len(parts):
            clause_text = parts[i+2].strip()
        combined_clauses.append(f"{number} {clause_text}")
        i += 3

    # Fallback: if no clause numbering is found, split by paragraphs
    if not combined_clauses:
        combined_clauses = [para.strip() for para in text.split('\n') if para.strip()]

    return combined_clauses

# Call the ChatCompletion API for semantic comparison
def analyze_clause(msa_clause, vendor_clause):
    # Create the chat messages
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an experienced attorney with expertise in contract law. "
                "Analyze discrepancies between two contract clauses."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Master Service Agreement (MSA) Clause:\n{msa_clause}\n\n"
                f"Vendor Contract Clause:\n{vendor_clause}\n\n"
                "Please provide a detailed analysis of any discrepancies and potential legal implications."
            )
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=messages,
            max_tokens=300,
            temperature=0.2, 
        )
        analysis = response.choices[0].message["content"].strip()
        return analysis
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# ------------------------------------------------------------
# 3. STREAMLIT APP
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Semantic Contract Comparator with AI", layout="wide")
    st.title("Master Service Agreement (MSA) vs. Vendor Contract Comparator with Semantic Analysis")
    
    st.markdown("""
    **Instructions**:
    1. Upload your Master Service Agreement (MSA) and the Vendor's Contract in PDF or DOCX formats.
    2. The application will segment the documents into clauses and perform a semantic comparison.
    3. Discrepancies will be highlighted with AI-generated explanations of potential legal implications.
    """)

    # File uploaders for MSA and Vendor contracts
    msa_file = st.file_uploader("Upload Master Service Agreement (MSA)", type=["pdf", "docx"])
    vendor_file = st.file_uploader("Upload Vendor Contract", type=["pdf", "docx"])
    
    if msa_file and vendor_file:
        with st.spinner('Processing documents and analyzing clauses...'):
            # Extract text from MSA
            if msa_file.type == "application/pdf":
                msa_text = extract_text_pdf(msa_file)
            elif msa_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                msa_text = extract_text_docx(msa_file)
            else:
                st.error("Unsupported MSA file format.")
                return

            # Extract text from Vendor Contract
            if vendor_file.type == "application/pdf":
                vendor_text = extract_text_pdf(vendor_file)
            elif vendor_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                vendor_text = extract_text_docx(vendor_file)
            else:
                st.error("Unsupported Vendor Contract file format.")
                return

            # Split documents into clauses
            msa_clauses = split_into_clauses(msa_text)
            vendor_clauses = split_into_clauses(vendor_text)

            # Perform semantic comparisons
            max_clauses = max(len(msa_clauses), len(vendor_clauses))
            discrepancies = []

            for i in range(max_clauses):
                msa_clause = msa_clauses[i] if i < len(msa_clauses) else ""
                vendor_clause = vendor_clauses[i] if i < len(vendor_clauses) else ""

                if msa_clause and vendor_clause:
                    # Compare clauses semantically using ChatCompletion
                    analysis = analyze_clause(msa_clause, vendor_clause)
                    discrepancies.append({
                        "Clause Number": f"Clause {i+1}",
                        "MSA Clause": msa_clause,
                        "Vendor Clause": vendor_clause,
                        "Analysis": analysis
                    })
                elif msa_clause and not vendor_clause:
                    discrepancies.append({
                        "Clause Number": f"Clause {i+1}",
                        "MSA Clause": msa_clause,
                        "Vendor Clause": "Missing in Vendor Contract",
                        "Analysis": "The vendor contract is missing this clause. This could lead to potential gaps in obligations and rights."
                    })
                elif vendor_clause and not msa_clause:
                    discrepancies.append({
                        "Clause Number": f"Clause {i+1}",
                        "MSA Clause": "Missing in MSA",
                        "Vendor Clause": vendor_clause,
                        "Analysis": "The MSA is missing this clause. This could introduce unexpected obligations or rights."
                    })

        st.success('Semantic Comparison Complete!')

        # Display Results
        st.header("Comparison Results")
        col1, col2, col3 = st.columns([2, 2, 5])

        with col1:
            st.subheader("Master Service Agreement (MSA) Clause")
        with col2:
            st.subheader("Vendor Contract Clause")
        with col3:
            st.subheader("Analysis & Discrepancies")

        for discrepancy in discrepancies:
            with col1:
                st.markdown(f"**{discrepancy['Clause Number']}**\n\n{discrepancy['MSA Clause']}")
            with col2:
                st.markdown(f"**{discrepancy['Clause Number']}**\n\n{discrepancy['Vendor Clause']}")
            with col3:
                st.markdown(f"**{discrepancy['Analysis']}**")
                st.markdown("---")

if __name__ == "__main__":
    main()
