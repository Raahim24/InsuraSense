import streamlit as st
import tempfile
import shutil
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

import fitz  # PyMuPDF
import google.generativeai as genai
import re
from pydantic import BaseModel, Field, ValidationError

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env or Streamlit secrets.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# --- Prompts (EXACT from notebook) ---
PROMPT_PA = """You are an expert medical document processing assistant specializing in Prior Authorization (PA) form analysis and field mapping. Your task is to process and enrich PA form field data with detailed contextual information.\n\nGiven Input:\n1. A structured dataset containing PA form field definitions including:\n   - Field names (e.g. CB1, T1)\n   - Field types (checkbox, text, etc.)\n   - Page numbers\n   - Field labels\n   - Current values\n\n2. The complete Prior Authorization form PDF document\n\nRequired Processing:\nFor each form field, analyze sequentially by page number and:\n\n1. Extract the implicit question being asked by the field\n   - For checkboxes: Frame the label as a yes/no question\n   - For text fields: Frame as an information request\n   - For dates: Specify what event/action the date refers to\n\n2. Generate rich contextual information that includes:\n   - The section/category the field belongs to\n   - Whether it's a primary question or sub-question\n   - Whose information is being requested (patient, provider, insurer)\n   - Any dependencies on other fields\n   - Clinical relevance of the requested information\n\n<CRITICAL_REQUIREMENTS>\n- Every field must have both question and context added\n- Context must be specific and clinically relevant\n- Maintain logical relationships between fields\n- Preserve exact field names and labels\n- Keep context concise but informative (25 words max)\n- Only output valid JSON\n</CRITICAL_REQUIREMENTS>\n\n<RESPONSE_FORMAT>\nEach output JSON object should only contain the fields - name, type, page, field_label, question, context in the following format:\n{{\"name\": \"CB1\",\n \"type\": \"checkbox\",\n \"page\": 2,\n \"field_label\": \"Start of treatment\",\n \"question\": \"Is this a new treatment start for the patient?\",\n \"context\": \"Initial checkbox in treatment timeline section indicating whether patient is beginning new therapy versus continuing existing treatment.\"}}\n{{\"name\": \"T2\",\n \"type\": \"text\", \n \"page\": 2,\n \"field_label\": \"Start date: (MM)\",\n \"question\": \"What is the month of treatment start?\",\n \"context\": \"2-digit month format for planned medication initiation date in treatment scheduling section.\"}}\n</RESPONSE_FORMAT>\n\n<PA_FORM_DATA>\n{page_fields}\n</PA_FORM_DATA>\n\nReturn valid JSON array only. No explanations outside the JSON."""

REFERRAL_PACKAGE_PROMPT = """You are an expert medical document processing assistant specializing in Prior Authorization (PA) forms and medical documentation. You are given a list of PA form fields with their associated context and questions. Your task is to thoroughly analyze the provided PDF referral package and extract all relevant information to accurately fill out the PA form.\n\n## CRITICAL INSTRUCTIONS:\n1. **NEVER leave answer fields empty or null** - always provide a specific value\n2. **For missing information**: Use \"Not documented\" or \"Not specified\" instead of empty strings\n3. **For checkbox fields**: Always answer with either \"Yes\" or \"No\" (never true/false or empty)\n4. **For text fields**: Provide the exact information or \"Not available\" if truly missing\n5. **For dates**: Use MM/DD/YYYY format (unless format is specified) or \"Not specified\" if date is missing \n6. **Be thorough**: Review the ENTIRE document multiple times to find all relevant information\n\n## DETAILED EXTRACTION GUIDELINES:\n\n### Patient Information:\n- Extract ALL demographic details (name, DOB, address, phone, insurance)\n- Look for patient information in headers, footers, cover pages, and forms\n- Check multiple pages for complete contact information\n\n### Medical Information:\n- **Diagnoses**: Extract primary and secondary diagnoses with ICD-10 codes if available\n- **Medications**: Include exact drug names, strengths, frequencies, routes of administration\n- **Treatment History**: Look for previous medications tried, dates, outcomes, failures\n- **Clinical Notes**: Extract relevant symptoms, assessments, lab results\n- **Provider Details**: Include all prescribing physicians, NPIs, addresses, phone numbers\n\n### Administrative Details:\n- **Insurance**: Member IDs, group numbers, prior authorization numbers\n- **Facility Information**: Infusion centers, pharmacies, administration locations\n- **Dates**: Treatment start dates, last treatment dates, prescription dates\n\n## ANSWER FORMAT REQUIREMENTS:\n\n**For Checkbox Fields (CB prefixes):**\n- Answer ONLY with \"Yes\" or \"No\" \n- If unclear, use clinical judgment based on available information\n- Example: If asking about \"Start of treatment\" and document shows new prescription → \"Yes\"\n\n**For Text Fields (T prefixes):**\n- Provide exact values from the document\n- For dates: Use MM/DD/YYYY format (e.g., \"05/22/2024\") unless format is specified \n- For names: Use exact spelling and format from document\n- For missing info: Use \"Not documented\" instead of leaving blank\n\n**For Yes/No Questions:**\n- Base answers on clinical evidence in the document\n- If patient has the condition/medication/history mentioned → \"Yes\"\n- If explicitly stated they don't have it or no evidence found → \"No\"\n\n## VALIDATION CHECKLIST:\nBefore submitting, ensure:\n- ✓ Every field has a non-empty answer\n- ✓ All checkbox answers are \"Yes\" or \"No\"\n- ✓ All dates follow MM/DD/YYYY format\n- ✓ Patient demographics are complete\n- ✓ Medication information is detailed and accurate\n- ✓ No fields are left with null, empty strings, or boolean values\n\n<PA_FORM_DATA>\n{pa_form_fields}\n</PA_FORM_DATA>\n\n<RESPONSE FORMAT>\n[\n  {{\n    \"name\": \"CB1\",\n    \"page\": 2,\n    \"field_label\": \"Start of treatment\",\n    \"answer\": \"Yes\"\n  }},\n  {{\n    \"name\": \"T2\",\n    \"page\": 2,\n    \"field_label\": \"Start date: (MM)\",\n    \"answer\": \"05\"\n  }}\n]\n</RESPONSE FORMAT>\n\n**CRITICAL**: Every field must have a specific answer - no empty strings, no null values, no boolean true/false."""

# --- Helper Functions (EXACT from notebook, with type hints and docstrings) ---
def extract_fields_with_positions(pdf_path: str) -> Dict[int, List[dict]]:
    """
    Extracts form fields and their positions from a PDF.
    Returns a dict of page_num -> list of field dicts.
    """
    doc = fitz.open(pdf_path)
    fields = []
    for page_num, page in enumerate(doc, start=1):
        for w in page.widgets() or []:
            field = {
                "name": w.field_name,
                "type": "checkbox" if w.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX else "text",
                "value": w.field_value,
                "page": page_num,
                "field_type": w.field_type,
                "field_type_string": w.field_type_string,
                "field_label": w.field_label,
            }
            fields.append(field)
    # Group by page
    fields_by_page = {}
    for field in fields:
        page_num = field['page']
        if page_num not in fields_by_page:
            fields_by_page[page_num] = []
        fields_by_page[page_num].append(field)
    return fields_by_page

async def query_gemini_async(prompt: str, pdf_path: str, model: str = "gemini-2.5-flash") -> str:
    filepath = Path(pdf_path)
    loop = asyncio.get_event_loop()
    generation_config = genai.GenerationConfig(response_mime_type="application/json")
    response = await loop.run_in_executor(
        None,
        lambda: genai.GenerativeModel(
            model,
            generation_config=generation_config
        ).generate_content([
            genai.upload_file(path=filepath),
            prompt
        ])
    )
    return response.text

async def process_pa_fields_async(pa_fields_data: Dict[int, List[dict]], pdf_path: str) -> Dict[int, Any]:
    async def process_page(page_num, page_fields):
        prompt = PROMPT_PA.format(page_fields=json.dumps(page_fields))
        result = await query_gemini_async(prompt, pdf_path)
        return page_num, result
    tasks = [process_page(page, fields) for page, fields in pa_fields_data.items()]
    results = await asyncio.gather(*tasks)
    enhanced_fields = {}
    for page, result in results:
        enhanced_fields[page] = result
    return enhanced_fields

class PAFormAnswer(BaseModel):
    name: str
    page: int
    field_label: str
    answer: str = Field(description="answer to the question based on the referral package PDF")

def _safe_json_loads(text: str) -> Any:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.S | re.I)
    if m:
        s = m.group(1).strip()
    a1, a2 = s.find('['), s.rfind(']')
    o1, o2 = s.find('{'), s.rfind('}')
    if a1 != -1 and a2 != -1 and a2 > a1:
        s = s[a1:a2+1]
    elif o1 != -1 and o2 != -1 and o2 > o1:
        s = s[o1:o2+1]
    return json.loads(s)

def parse_and_validate_answers(response_text: str) -> List[PAFormAnswer]:
    data = _safe_json_loads(response_text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array or object.")
    validated: List[PAFormAnswer] = []
    for i, item in enumerate(data):
        try:
            validated.append(PAFormAnswer(**item))
        except ValidationError as e:
            raise ValueError(f"Pydantic validation failed at index {i}: {e}") from e
    return validated

async def fill_single_page_from_referral(page_num: int,
                                         page_fields: list,
                                         referral_pdf_path: str,
                                         model: str = "gemini-2.5-flash"):
    prompt = REFERRAL_PACKAGE_PROMPT.format(
        pa_form_fields=json.dumps(page_fields, indent=2)
    )
    loop = asyncio.get_event_loop()
    uploaded = await loop.run_in_executor(
        None, lambda: genai.upload_file(path=Path(referral_pdf_path))
    )
    response = await loop.run_in_executor(
        None,
        lambda: genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            ),
        ).generate_content([uploaded, prompt]),
    )
    answers = parse_and_validate_answers(response.text)
    return page_num, [a.model_dump() for a in answers]

async def fill_pa_pages_sequential(enhanced_fields_by_page: Dict,
                                   referral_pdf_path: str,
                                   model: str = "gemini-2.5-flash"):
    first_key = next(iter(enhanced_fields_by_page), None)
    if first_key is None:
        return {}
    if isinstance(enhanced_fields_by_page[first_key], dict):
        by_page: Dict[int, list] = {}
        for f in enhanced_fields_by_page.values():
            p = int(f["page"])
            by_page.setdefault(p, []).append(f)
        enhanced_fields_by_page = by_page
    filled_pages: Dict[int, List[Dict[str, Any]]] = {}
    for page_num in sorted(enhanced_fields_by_page.keys()):
        page_fields = enhanced_fields_by_page[page_num]
        page_ret, page_results = await fill_single_page_from_referral(
            page_num=page_num,
            page_fields=page_fields,
            referral_pdf_path=referral_pdf_path,
            model=model,
        )
        filled_pages[page_ret] = page_results
    return filled_pages

def build_answer_index(filled_results: dict) -> dict:
    idx = {}
    for page, items in filled_results.items():
        for it in items:
            name = str(it.get("name", "")).strip()
            if not name:
                continue
            idx[name] = str(it.get("answer", "")).strip()
    return idx

def _bool_from_yes_no(s: str) -> bool:
    return str(s).strip().lower() in ("yes", "y", "true", "checked", "on", "1")

def fill_pa_pdf_from_answers(
    pa_pdf_in: str,
    filled_results: dict,
    out_pdf: str = None,
    make_flattened_copy: bool = False
) -> str:
    pa_pdf_in = str(pa_pdf_in)
    out_pdf = out_pdf or str(Path(pa_pdf_in).with_name(Path(pa_pdf_in).stem + "_filled.pdf"))
    answers_by_name = build_answer_index(filled_results)
    doc = fitz.open(pa_pdf_in)
    filled = 0
    missing = []
    for page in doc:
        widgets = page.widgets() or []
        for w in widgets:
            fname = w.field_name or ""
            if not fname:
                continue
            if fname not in answers_by_name:
                missing.append(fname)
                continue
            ans = answers_by_name[fname]
            try:
                if w.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                    checked = _bool_from_yes_no(ans)
                    w.field_value = "Yes" if checked else "Off"
                    w.update()
                    filled += 1
                else:
                    w.field_value = ans
                    w.update()
                    filled += 1
            except Exception:
                try:
                    if w.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                        w.button_set(_bool_from_yes_no(ans))
                    else:
                        w.set_value(str(ans))
                    w.update()
                    filled += 1
                except Exception as e:
                    print(f"Could not write field '{fname}': {e}")
    doc.save(out_pdf, deflate=True)
    doc.close()
    if make_flattened_copy:
        flat_path = str(Path(out_pdf).with_name(Path(out_pdf).stem + "_flat.pdf"))
        d2 = fitz.open(out_pdf)
        d2.save(flat_path, deflate=True, garbage=4, clean=True)
        d2.close()
        return out_pdf, flat_path
    return out_pdf, None

# --- Streamlit UI ---
st.set_page_config(
    page_title="InsuraSense - PA Form Auto-Filler",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for light, modern theme
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card-like containers */
    .upload-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #bdc3c7;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc8 100%);
        color: #2e7d32;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    
    .stError {
        background: linear-gradient(135deg, #ffcdd2 0%, #f8bbd9 100%);
        color: #c62828;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    
    /* Info message styling */
    .stInfo {
        background: linear-gradient(135deg, #e1f5fe 0%, #e8f4f8 100%);
        color: #01579b;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(76, 175, 80, 0.4);
    }
    
    /* Spinner styling */
    .stSpinner {
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Feature icons */
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header with enhanced styling
st.markdown('<h1 class="main-title">InsuraSense</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Prior Authorization Form Auto-Filler</p>', unsafe_allow_html=True)

# Features section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h4>AI-Powered</h4>
        <p>Advanced AI analyzes your documents intelligently</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h4>Fast Processing</h4>
        <p>Get your filled PA forms in minutes, not hours</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h4>High Accuracy</h4>
        <p>Precise field mapping and medical data extraction</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Upload form with enhanced styling


with st.form("form-upload", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Blank PA Form**")
        pa_pdf = st.file_uploader(
            "Choose your blank Prior Authorization form", 
            type=["pdf"], 
            key="pa_pdf",
            help="Upload the empty PA form that needs to be filled out"
        )
        if pa_pdf:
            st.success(f"PA form uploaded: {pa_pdf.name}")
    
    with col2:
        st.markdown("**Referral Package**")
        referral_pdf = st.file_uploader(
            "Choose your medical referral package", 
            type=["pdf"], 
            key="referral_pdf",
            help="Upload the medical documents containing patient information"
        )
        if referral_pdf:
            st.success(f"Referral package uploaded: {referral_pdf.name}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.form_submit_button(
            "Generate Filled PA Form",
            use_container_width=True,
            disabled=not (pa_pdf and referral_pdf)
        )
        
    if not (pa_pdf and referral_pdf) and st.session_state.get('form_attempted', False):
        st.warning("Please upload both documents to continue")

if submit and pa_pdf and referral_pdf:
    # Add form attempted flag
    st.session_state['form_attempted'] = True
    
    # Enhanced processing with progress indicators
    progress_container = st.container()
    with progress_container:
        st.markdown("""
        <div class="upload-card" style="text-align: center;">
            <h3 style="color: #2c3e50;">Processing Your Documents</h3>
            <p style="color: #7f8c8d;">Please wait while our AI analyzes and fills your PA form...</p>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Step 1: Save uploaded files
                status_text.text("Saving uploaded files...")
                progress_bar.progress(10)
                
                pa_path = Path(tmpdir) / "pa.pdf"
                referral_path = Path(tmpdir) / "referral.pdf"
                with open(pa_path, "wb") as f:
                    f.write(pa_pdf.read())
                with open(referral_path, "wb") as f:
                    f.write(referral_pdf.read())
                
                # Step 2: Extract fields
                status_text.text("Extracting form fields...")
                progress_bar.progress(25)
                pa_fields = extract_fields_with_positions(str(pa_path))
                
                # Step 3: Generate enhanced fields
                status_text.text("AI analyzing form structure...")
                progress_bar.progress(50)
                enhanced_fields = asyncio.run(process_pa_fields_async(pa_fields, str(pa_path)))
                enhanced_fields_by_page = {p: json.loads(j) for p, j in enhanced_fields.items()}
                
                # Step 4: Fill answers from referral package
                status_text.text("AI extracting medical information...")
                progress_bar.progress(75)
                filled_results = asyncio.run(fill_pa_pages_sequential(enhanced_fields_by_page, str(referral_path)))
                
                # Step 5: Fill the PA PDF
                status_text.text("Generating filled PA form...")
                progress_bar.progress(90)
                filled_pdf_path, flattened_path = fill_pa_pdf_from_answers(
                    str(pa_path), 
                    filled_results, 
                    out_pdf=str(Path(tmpdir)/"PA_filled.pdf"), 
                    make_flattened_copy=True
                )
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Success message and download section
                st.success("Your PA Form is Ready!")
                st.markdown("""
                <div class="upload-card" style="text-align: center; background: linear-gradient(135deg, #a8e6cf 0%, #dcedc8 100%);">
                    <h3 style="color: #2e7d32;">Your PA Form is Ready!</h3>
                    <p style="color: #388e3c;">Download your completed forms below</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Download buttons in columns
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with open(filled_pdf_path, "rb") as f:
                        st.download_button(
                            "Download Filled PA Form",
                            f.read(),
                            file_name="PA_filled.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    if flattened_path:
                        with open(flattened_path, "rb") as f:
                            st.download_button(
                                "Download Flattened Version",
                                f.read(),
                                file_name="PA_filled_flat.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
        except Exception as e:
            logger.error(f"Error: {e}")
            st.markdown("""
            <div class="upload-card" style="background: linear-gradient(135deg, #ffcdd2 0%, #f8bbd9 100%); text-align: center;">
                <h3 style="color: #c62828;">❌ Processing Error</h3>
                <p style="color: #d32f2f;">Something went wrong during processing. Please try again.</p>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"Error details: {e}")
            
# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p><strong>InsuraSense</strong> - Streamlining healthcare administration with AI</p>
    <p style="font-size: 0.9rem;">Built with Streamlit and AI Technology</p>
</div>
""", unsafe_allow_html=True)
