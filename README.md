![](https://valenta.io/wp-content/uploads/2023/08/IDP-Services-1.jpg)
# Document Intelligence: Advanced Contract Analysis and Chat Application

## App Link: https://docuintel.streamlit.app/
## Overview
Document Intelligence is a powerful tool designed for advanced contract analysis and interaction. Leveraging the Gemini API, it supports PDF, DOCX, and TXT files to provide comprehensive contract insights through a user-friendly Streamlit interface. The application offers semantic clause comparison, regulatory compliance checks, contract quality assessment, executive summaries, and contextual Q&A capabilities.

## Features
1. **Semantic Clause Comparison**  
   - Matches clauses between contracts (e.g., MSA and Vendor) using semantic similarity.  
   - Highlights differences, risks, and provides suggestions for non-exact matches.

2. **Regulatory Compliance Check**  
   - Identifies missing or non-compliant clauses based on standard frameworks (e.g., GDPR, UCC).

3. **Contract Compliance & Quality Assessment**  
   - Evaluates vendor contracts against the MSA for clause coverage.  
   - Assesses clarity, completeness, and enforceability of contracts.

4. **Executive Summary**  
   - Generates a high-level overview of findings, including key differences, risks, compliance issues, and contract quality.

5. **Contextual Q&A**  
   - Answers user queries about contracts using semantic search to locate relevant clauses.

6. **Sample Contract Loading**  
   - Includes sample contracts for testing and demonstration purposes.

## Tech Stack
- **Backend**: Python, Gemini API (for embeddings and content generation)  
- **Frontend**: Streamlit for a professional, interactive UI  
- **File Processing**: PyPDF2 (PDF), python-docx (DOCX), and text handling for TXT files  
- **Semantic Analysis**: Embeddings for semantic matching, cosine similarity for relevance scoring  
- **Error Handling**: Robust error management and progress tracking for reliable performance

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abh2050/docanalyzer.git
   cd document-intelligence
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your Gemini API key:
     ```env
     GEMINI_API_KEY=your-api-key
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Upload Contracts**:
   - Use the Streamlit UI to upload MSA and Vendor contracts in PDF, DOCX, or TXT format.
   - Alternatively, load sample contracts for testing.

2. **Analyze Contracts**:
   - Click "Evaluate Contract & Generate Summary" to perform compliance checks, quality assessment, and generate an executive summary.
   - Download the executive summary as a TXT file.

3. **Ask Questions**:
   - Use the "Contract Q&A" tab to query specific details about uploaded contracts.
   - Get AI-powered answers based on semantic search.

## Directory Structure
```plaintext
document-intelligence/
├── app.py                # Main Streamlit application
├── examples/             # Sample contracts (e.g., MSA.docx, Vendor.docx)
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (API keys)
└── README.md             # This file
```

## Dependencies
Listed in `requirements.txt`. Key libraries include:
- `streamlit`
- `google-generativeai`
- `PyPDF2`
- `python-docx`
- `numpy`
- `python-dotenv`

## Notes
- Ensure a valid Gemini API key is configured in the `.env` file.
- The application handles large contracts by chunking text for embeddings to avoid API limits.
- Sample contracts are provided in the `examples/` folder for quick testing.
- Disclaimer: AI-generated outputs are for informational purposes and should not be considered legal advice. Consult qualified legal counsel for professional guidance.

## Example Usecase
![](https://github.com/abh2050/docanalyzer/blob/main/samples/pic1.png)
![](https://github.com/abh2050/docanalyzer/blob/main/samples/pic2.png)
![](https://github.com/abh2050/docanalyzer/blob/main/samples/pic4.png)
![](https://github.com/abh2050/docanalyzer/blob/main/samples/pic3.png)

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

```

This README provides a clear and comprehensive guide to the Document Intelligence application, covering its features, setup, usage, and contribution guidelines. It’s structured to be user-friendly for both developers and end-users. Let me know if you need any adjustments!
