# llm_service.py
import json
import asyncio
import google.generativeai as genai
import os
import pathlib

# Load environment variables (for API key)
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Prompt for enhancing PA form fields
PROMPT_PA_ENHANCEMENT = """You are an expert medical document processing assistant specializing in Prior Authorization (PA) form analysis and field mapping. Your task is to process and enrich PA form field data with detailed contextual information.

Given Input:
1. A structured dataset containing PA form field definitions including:
   - Field names (e.g. CB1, T1)
   - Field types (checkbox, text, etc.)
   - Page numbers
   - Field labels
   - Current values

2. The complete Prior Authorization form PDF document

Required Processing:
For each form field, analyze sequentially by page number and:

1. Extract the implicit question being asked by the field
   - For checkboxes: Frame the label as a yes/no question
   - For text fields: Frame as an information request
   - For dates: Specify what event/action the date refers to

2. Generate rich contextual information that includes:
   - The section/category the field belongs to
   - Whether it's a primary question or sub-question
   - Whose information is being requested (patient, provider, insurer)
   - Any dependencies on other fields
   - Clinical relevance of the requested information

<CRITICAL_REQUIREMENTS>
- Every field must have both question and context added
- Context must be specific and clinically relevant
- Maintain logical relationships between fields
- Preserve exact field names and labels
- Keep context concise but informative (25 words max)
- Only output valid JSON
</CRITICAL_REQUIREMENTS>

<RESPONSE_FORMAT>
Each output JSON object should only contain the fields - name, type, page, field_label, question, context in the following format:
{{"name": "CB1",
 "type": "checkbox",
 "page": 2,
 "field_label": "Start of treatment",
 "question": "Is this a new treatment start for the patient?",
 "context": "Initial checkbox in treatment timeline section indicating whether patient is beginning new therapy versus continuing existing treatment."}}
{{"name": "T2",
 "type": "text", 
 "page": 2,
 "field_label": "Start date: (MM)",
 "question": "What is the month of treatment start?",
 "context": "2-digit month format for planned medication initiation date in treatment scheduling section."}}
</RESPONSE_FORMAT>

<PA_FORM_DATA>
{page_fields}
</PA_FORM_DATA>

Return valid JSON array only. No explanations outside the JSON."""

# Prompt for filling PA form fields from referral package
PROMPT_REFERRAL_PACKAGE_FILLING = """You are an expert medical document processing assistant specializing in Prior Authorization (PA) forms and medical documentation. You are given a list of PA form fields with their associated context and questions. Your task is to thoroughly analyze the provided PDF referral package and extract all relevant information to accurately fill out the PA form.

## CRITICAL INSTRUCTIONS:
1. **NEVER leave answer fields empty or null** - always provide a specific value
2. **For missing information**: Use "Not documented" or "Not specified" instead of empty strings
3. **For checkbox fields**: Always answer with either "Yes" or "No" (never true/false or empty)
4. **For text fields**: Provide the exact information or "Not available" if truly missing
5. **For dates**: Use MM/DD/YYYY format (unless format is specified) or "Not specified" if date is missing 
6. **Be thorough**: Review the ENTIRE document multiple times to find all relevant information

## DETAILED EXTRACTION GUIDELINES:

### Patient Information:
- Extract ALL demographic details (name, DOB, address, phone, insurance)
- Look for patient information in headers, footers, cover pages, and forms
- Check multiple pages for complete contact information

### Medical Information:
- **Diagnoses**: Extract primary and secondary diagnoses with ICD-10 codes if available
- **Medications**: Include exact drug names, strengths, frequencies, routes of administration
- **Treatment History**: Look for previous medications tried, dates, outcomes, failures
- **Clinical Notes**: Extract relevant symptoms, assessments, lab results
- **Provider Details**: Include all prescribing physicians, NPIs, addresses, phone numbers

### Administrative Details:
- **Insurance**: Member IDs, group numbers, prior authorization numbers
- **Facility Information**: Infusion centers, pharmacies, administration locations
- **Dates**: Treatment start dates, last treatment dates, prescription dates

## ANSWER FORMAT REQUIREMENTS:

**For Checkbox Fields (CB prefixes):**
- Answer ONLY with "Yes" or "No" 
- If unclear, use clinical judgment based on available information
- Example: If asking about "Start of treatment" and document shows new prescription → "Yes"

**For Text Fields (T prefixes):**
- Provide exact values from the document
- For dates: Use MM/DD/YYYY format (e.g., "05/22/2024") unless format is specified 
- For names: Use exact spelling and format from document
- For missing info: Use "Not documented" instead of leaving blank

**For Yes/No Questions:**
- Base answers on clinical evidence in the document
- If patient has the condition/medication/history mentioned → "Yes"
- If explicitly stated they don't have it or no evidence found → "No"

## VALIDATION CHECKLIST:
Before submitting, ensure:
- ✓ Every field has a non-empty answer
- ✓ All checkbox answers are "Yes" or "No"
- ✓ All dates follow MM/DD/YYYY format
- ✓ Patient demographics are complete
- ✓ Medication information is detailed and accurate
- ✓ No fields are left with null, empty strings, or boolean values

<PA_FORM_DATA>
{pa_form_fields}
</PA_FORM_DATA>

<RESPONSE FORMAT>
Return ONLY a JSON array with this exact structure for each field:
[
  {{
    "name": "CB1",
    "page": 2,
    "field_label": "Start of treatment",
    "answer": "Yes"
  }},
  {{
    "name": "T2",
    "page": 2,
    "field_label": "Start date: (MM)",
    "answer": "05"
  }}
]
</RESPONSE FORMAT>

**CRITICAL**: Every field must have a specific answer - no empty strings, no null values, no boolean true/false."""

async def query_gemini_with_pdf_async(prompt, pdf_bytes, model="gemini-2.5-flash"):
    """
    Sends a prompt and a PDF (as bytes) to the Gemini API and returns the text response.
    """
    # Create a temporary file to upload the PDF bytes
    temp_pdf_path = "temp_uploaded_pdf.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    try:
        filepath = pathlib.Path(temp_pdf_path)
        loop = asyncio.get_event_loop()
        
        # Define generation_config here
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )
        
        response = await loop.run_in_executor(
            None,
            lambda: genai.GenerativeModel(model).generate_content(
                [genai.upload_file(path=filepath), prompt],
                generation_config=generation_config # Pass generation_config here
            )
        )
        return response.text
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

async def enhance_pa_fields(pa_fields_data, pa_pdf_bytes):
    """
    Enhances PA form fields with questions and context using Gemini.

    Args:
        pa_fields_data (dict): Dictionary of PA fields grouped by page.
        pa_pdf_bytes (bytes): The content of the PA PDF file as bytes.

    Returns:
        dict: Enhanced PA fields grouped by page.
    """
    async def process_page(page_num, page_fields):
        prompt = PROMPT_PA_ENHANCEMENT.format(page_fields=json.dumps(page_fields))
        result = await query_gemini_with_pdf_async(prompt, pa_pdf_bytes)
        return page_num, result
    
    tasks = [process_page(page, fields) for page, fields in pa_fields_data.items()]
    results = await asyncio.gather(*tasks)
    
    enhanced_fields = {}
    for page, result_json_str in results:
        try:
            enhanced_fields[page] = json.loads(result_json_str)
            print(f"Page {page} enhanced successfully.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for page {page}: {e}")
            print(f"Raw response for page {page}: {result_json_str}")
            enhanced_fields[page] = [] # Return empty list or handle error as needed
    
    return enhanced_fields

async def fill_pa_form_from_referral(enhanced_fields_by_page, referral_pdf_bytes):
    """
    Fills PA form fields by extracting information from a referral PDF using Gemini.

    Args:
        enhanced_fields_by_page (dict): Enhanced PA fields grouped by page.
        referral_pdf_bytes (bytes): The content of the referral PDF file as bytes.

    Returns:
        dict: Filled PA fields grouped by page.
    """
    async def fill_page(page_num, page_fields):
        prompt = PROMPT_REFERRAL_PACKAGE_FILLING.format(
            pa_form_fields=json.dumps(page_fields, indent=2)
        )
        result = await query_gemini_with_pdf_async(prompt, referral_pdf_bytes)
        return page_num, result
    
    tasks = [
        fill_page(page, fields)
        for page, fields in enhanced_fields_by_page.items() # Corrected from enhanced_fields_by_items()
    ]
    
    results = await asyncio.gather(*tasks)
    
    filled_pages = {}
    for page_num, page_results_json_str in results:
        try:
            filled_pages[page_num] = json.loads(page_results_json_str)
            print(f"Page {page_num} filled successfully.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for filled page {page_num}: {e}")
            print(f"Raw response for filled page {page_num}: {page_results_json_str}")
            filled_pages[page_num] = [] # Handle error
    
    return filled_pages
