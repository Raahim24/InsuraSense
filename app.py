# streamlit_app.py
import streamlit as st
import asyncio
import json
import os
from pdf_utils import extract_fields_with_positions, fill_pdf_form
from llm_service import enhance_pa_fields, fill_pa_form_from_referral

st.set_page_config(layout="wide", page_title="PA Form Autofill AI")

def display_message(message, type="info"):
    """Displays a styled message."""
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "error":
        st.error(message)
    elif type == "warning":
        st.warning(message)

async def main_app():
    st.title("📄 AI-Powered Prior Authorization Form Autofill")
    st.markdown("""
        Upload your Prior Authorization (PA) form and the patient's referral package PDF. 
        Our AI will extract relevant information from the referral and automatically fill out the PA form for you.
        ---
    """)

    # File Uploaders
    col1, col2 = st.columns(2)
    with col1:
        pa_form_file = st.file_uploader("Upload PA Form (PDF)", type=["pdf"], key="pa_uploader")
    with col2:
        referral_package_file = st.file_uploader("Upload Referral Package (PDF)", type=["pdf"], key="referral_uploader")

    filled_pdf_bytes = None

    if pa_form_file and referral_package_file:
        st.markdown("---")
        display_message("Processing your documents...", "info")

        pa_pdf_bytes = pa_form_file.read()
        referral_pdf_bytes = referral_package_file.read()

        try:
            with st.spinner("Step 1/3: Extracting PA form fields..."):
                # Extract fields from PA form
                # We need a temporary file path for fitz to open from bytes
                temp_pa_path = "temp_pa_form.pdf"
                with open(temp_pa_path, "wb") as f:
                    f.write(pa_pdf_bytes)
                pa_fields_data = extract_fields_with_positions(temp_pa_path)
                os.remove(temp_pa_path) # Clean up temp file
                display_message(f"Found {sum(len(v) for v in pa_fields_data.values())} fields in the PA form.", "success")

            if not pa_fields_data:
                display_message("No fillable fields found in the PA form. Please ensure it's a fillable PDF.", "warning")
                return

            with st.spinner("Step 2/3: Enhancing PA form fields with AI context..."):
                # Enhance PA fields using Gemini
                enhanced_pa_data = await enhance_pa_fields(pa_fields_data, pa_pdf_bytes)
                display_message("PA form fields enhanced with AI context.", "success")

            with st.spinner("Step 3/3: Filling PA form from referral package with AI..."):
                # Fill PA form from referral package using Gemini
                filled_results_by_page = await fill_pa_form_from_referral(enhanced_pa_data, referral_pdf_bytes)
                
                # Flatten the filled results for easier mapping to the PDF
                flattened_filled_results = {}
                for page_num, page_fields in filled_results_by_page.items():
                    for field in page_fields:
                        flattened_filled_results[field['name']] = field # Use field name as key

                display_message("PA form fields filled from referral package.", "success")

            with st.spinner("Finalizing PDF..."):
                # Fill the actual PDF
                filled_pdf_bytes = fill_pdf_form(pa_pdf_bytes, flattened_filled_results)
                display_message("PA form successfully autofilled!", "success")

            if filled_pdf_bytes:
                st.download_button(
                    label="Download Filled PA Form",
                    data=filled_pdf_bytes,
                    file_name="filled_PA_form.pdf",
                    mime="application/pdf",
                    help="Click to download your automatically filled Prior Authorization form."
                )
                display_message("Your filled PDF is ready for download.", "info")

        except Exception as e:
            display_message(f"An error occurred during processing: {e}", "error")
            st.exception(e) # Display full traceback for debugging

    else:
        display_message("Please upload both the PA Form and the Referral Package PDFs to begin.", "info")

if __name__ == "__main__":
    # Ensure asyncio is running for async functions
    if 'GEMINI_API_KEY' not in os.environ:
        st.error("GEMINI_API_KEY environment variable not set. Please set it to run the application.")
    else:
        asyncio.run(main_app())
