# InsuraSense: Automated Prior Authorization Form Filling for Healthcare

---

## Overview

InsuraSense is an end-to-end automation pipeline designed to streamline the Prior Authorization (PA) process in healthcare. By leveraging advanced AI, OCR, and PDF processing, InsuraSense extracts critical clinical and administrative information from scanned referral packages and automatically populates structured PA forms for insurance approval. This solution reduces manual effort, minimizes errors, and accelerates patient access to necessary treatments.

---

## Motivation

Prior Authorization is a time-consuming, error-prone process that requires healthcare staff to manually review referral documents and transcribe information into complex insurance forms. InsuraSense automates this workflow, enabling:
- Faster insurance approvals
- Reduced administrative burden
- Improved accuracy and compliance
- Scalable handling of diverse form types and patient cases

---

## Key Features

- **Automated Field Extraction:** Identifies and contextualizes all fillable fields (text, checkboxes) in PA forms, supporting both standard and custom layouts.
- **AI-Powered Data Extraction:** Uses OCR and large language models to extract, validate, and map relevant information from scanned referral packages.
- **Intelligent Form Filling:** Populates PA PDFs with extracted answers, handling checkboxes, text, and date fields with clinical logic and formatting rules.
- **Missing Information Reporting:** Generates a clear report for each patient, listing any required fields that could not be populated from the referral package.
- **Flexible Output:** Produces both editable and flattened (non-editable) filled PDFs for downstream use.
- **Modular, Extensible Design:** Built for easy adaptation to new form types, drugs, and insurance requirements.

---

## Workflow

1. **Input Preparation:**
   - Organize patient data in the `Input Data/` directory, with each patient having a PA form PDF and a referral package PDF (scanned documents).

2. **Form Field Extraction:**
   - The pipeline parses the PA form, extracting all fillable fields and their metadata (name, type, label, page, etc.).

3. **Contextual Field Analysis:**
   - An AI model generates human-readable questions and clinical context for each field, enabling robust mapping from unstructured documents.

4. **Referral Package Processing:**
   - OCR and AI extract answers for each form field from the referral package, following strict clinical and formatting guidelines.

5. **Form Filling:**
   - The extracted answers are written into the PA PDF, supporting both text and checkbox fields. Both editable and flattened versions are saved.

6. **Reporting:**
   - For each patient, a report is generated listing any fields for which information was missing or not documented in the referral package.

---

## Directory Structure

```
InsuraSense/
  ├── app.py                    # Main pipeline and codebase
  ├── Input Data/               # Input data organized by patient
  │     ├── PatientA/
  │     │     ├── PA.pdf
  │     │     └── referral_package.pdf
  │     └── ...
  ├── info/                     # Extracted data and intermediate outputs
  ├── output_examples/          # (Recommended) Example filled forms and reports
  └── README.md                 # Project documentation
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/InsuraSense.git
   cd InsuraSense
   ```
2. **Install dependencies:**
   - Python 3.9+
   - Recommended: Create a virtual environment
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
3. **Set up API keys:**
   - Place your Gemini API key in a `.env` file as `GEMINI_API_KEY=your_key_here`

---

## Usage

1. **Prepare Input Data:**
   - Place each patient's PA form and referral package in a dedicated subfolder under `Input Data/`.

2. **Run the Pipeline:**
   - Open and execute `app.py` in VSCode.
   - The notebook will:
     - Extract and contextualize form fields
     - Process referral packages
     - Fill the PA PDF and generate reports
     - Save outputs in the corresponding patient folder

3. **Review Outputs:**
   - Filled PA forms: `Input Data/<Patient>/PA_filled.pdf` 
   - Missing info reports: (customizable, e.g., `Input Data/<Patient>/missing_fields.md`)

---

## Example Output

- **Filled PA Form:** See `Input Data/Abdulla/PA_filled.pdf`

---

## Assumptions & Limitations

- The pipeline is optimized for widget-based (fillable) PDF forms. Non-widget forms may require additional handling.
- Referral packages must be high-quality scans for best OCR results.
- The AI model is tuned for English-language, US healthcare forms.
- Some fields may remain unfilled if information is not present in the referral package.

---

## Future Work

- Support for non-widget (non-interactive) PDF forms
- Enhanced OCR for low-quality scans
- Customizable output formats (e.g., CSV, JSON reports)
- Integration with EHR and insurance APIs
- User interface for manual review and correction

---

   - PA forms come in two formats: interactive widget-based PDFs (containing AcroForm widgets) and non-widget-based PDFs.
   - The primary expectation is for the pipeline to work with widget-based PDFs that contain fillable form fields.
   - While the solution should be designed to handle any form type, successfully implementing support for non-widget-based PDFs will be considered a bonus achievement.
   - The solution should prioritize robust handling of interactive widget-based forms first, then extend capabilities to non-widget formats if possible.

