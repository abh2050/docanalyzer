import streamlit as st
import openai
import os
from dotenv import load_dotenv
import nltk
import re
import numpy as np
from numpy.linalg import norm
import docx
import PyPDF2
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

# Download NLTK data at startup
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class ComparisonResult:
    index: int
    msa_text: str
    vendor_text: str
    similarity_score: float
    analysis: str

class DocumentProcessor:
    @staticmethod
    def extract_text_docx(file) -> str:
        try:
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error processing DOCX file: {str(e)}")
            return ""

    @staticmethod
    def extract_text_pdf(file) -> str:
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page_text := page.extract_text():
                    text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing PDF file: {str(e)}")
            return ""

    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        # Improved paragraph splitting with better handling of formatting
        text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [p for p in paragraphs if len(p.split()) > 5]  # Filter out very short paragraphs

class EmbeddingProcessor:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.cache = {}

    def get_embedding(self, text: str) -> Optional[List[float]]:
        # Cache embeddings to reduce API calls
        if text in self.cache:
            return self.cache[text]

        try:
            time.sleep(0.1)  # Rate limiting
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            embedding = response["data"][0]["embedding"]
            self.cache[text] = embedding
            return embedding
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return None

    @staticmethod
    def cosine_similarity(vecA: List[float], vecB: List[float]) -> float:
        vecA = np.array(vecA)
        vecB = np.array(vecB)
        return np.dot(vecA, vecB) / (norm(vecA) * norm(vecB))

class ContractAnalyzer:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def analyze_clause(self, msa_paragraph: str, vendor_paragraph: str) -> str:
        messages = [
            {"role": "system", 
             "content": """You are an expert contract analyst. Compare the following clauses and provide a concise analysis focusing on:
             1. Key differences in meaning or requirements
             2. Potential risks or concerns
             3. Specific recommendations
             Be brief and direct."""},
            {"role": "user", 
             "content": f"MSA Clause:\n{msa_paragraph}\n\nVendor Clause:\n{vendor_paragraph}"}
        ]
        
        try:
            time.sleep(0.1)  # Rate limiting
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"Analysis error: {str(e)}"

def main():
    st.set_page_config(page_title="Contract Comparison Tool", layout="wide")
    
    st.title("MSA vs. Vendor Contract Analyzer")
    st.markdown("""
    ### How it works
    1. Upload your MSA and vendor contract (PDF or DOCX)
    2. The tool splits documents into meaningful sections
    3. AI identifies matching clauses using semantic similarity
    4. Each pair is analyzed for key differences and potential issues
    """)

    # File uploaders with clear instructions
    col1, col2 = st.columns(2)
    with col1:
        msa_file = st.file_uploader("Upload MSA (PDF/DOCX)", type=["pdf", "docx"])
    with col2:
        vendor_file = st.file_uploader("Upload Vendor Contract (PDF/DOCX)", type=["pdf", "docx"])

    if not (msa_file and vendor_file):
        st.info("Please upload both documents to begin analysis")
        return

    doc_processor = DocumentProcessor()
    embedding_processor = EmbeddingProcessor()
    contract_analyzer = ContractAnalyzer()

    with st.spinner("Processing documents..."):
        # Extract text based on file type
        msa_text = (doc_processor.extract_text_pdf(msa_file) if msa_file.type == "application/pdf" 
                   else doc_processor.extract_text_docx(msa_file))
        vendor_text = (doc_processor.extract_text_pdf(vendor_file) if vendor_file.type == "application/pdf"
                      else doc_processor.extract_text_docx(vendor_file))

        if not (msa_text and vendor_text):
            st.error("Could not extract text from one or both documents")
            return

        # Process documents
        msa_paragraphs = doc_processor.split_into_paragraphs(msa_text)
        vendor_paragraphs = doc_processor.split_into_paragraphs(vendor_text)

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Compute embeddings with progress tracking
        total_paragraphs = len(msa_paragraphs) + len(vendor_paragraphs)
        current_progress = 0

        msa_embeds = []
        for i, p in enumerate(msa_paragraphs):
            status_text.text(f"Processing MSA paragraph {i+1}/{len(msa_paragraphs)}")
            emb = embedding_processor.get_embedding(p)
            msa_embeds.append(emb)
            current_progress += 1
            progress_bar.progress(current_progress / total_paragraphs)

        vendor_embeds = []
        for i, p in enumerate(vendor_paragraphs):
            status_text.text(f"Processing vendor paragraph {i+1}/{len(vendor_paragraphs)}")
            emb = embedding_processor.get_embedding(p)
            vendor_embeds.append(emb)
            current_progress += 1
            progress_bar.progress(current_progress / total_paragraphs)

        # Match paragraphs
        results = []
        status_text.text("Analyzing matches...")
        
        similarity_threshold = 0.7  # Adjust as needed
        for i, msa_emb in enumerate(msa_embeds):
            if msa_emb is None:
                continue
                
            best_match = max(
                ((j, embedding_processor.cosine_similarity(msa_emb, vendor_emb))
                 for j, vendor_emb in enumerate(vendor_embeds)
                 if vendor_emb is not None),
                key=lambda x: x[1],
                default=(None, 0)
            )
            
            if best_match[0] is not None and best_match[1] >= similarity_threshold:
                analysis = contract_analyzer.analyze_clause(
                    msa_paragraphs[i],
                    vendor_paragraphs[best_match[0]]
                )
                
                results.append(ComparisonResult(
                    index=i+1,
                    msa_text=msa_paragraphs[i],
                    vendor_text=vendor_paragraphs[best_match[0]],
                    similarity_score=best_match[1],
                    analysis=analysis
                ))

        progress_bar.empty()
        status_text.empty()

    # Display results in an organized table
    st.header("Comparison Results")
    
    for result in results:
        with st.expander(f"Clause Pair {result.index} (Similarity: {result.similarity_score:.2f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**MSA Clause**")
                st.write(result.msa_text)
            with col2:
                st.markdown("**Vendor Clause**")
                st.write(result.vendor_text)
            
            st.markdown("**Analysis**")
            st.write(result.analysis)

if __name__ == "__main__":
    main()
