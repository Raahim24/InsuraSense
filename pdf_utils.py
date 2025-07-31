# pdf_utils.py
import fitz
from io import BytesIO

def extract_fields_with_positions(pdf_path):
    """
    Extracts form fields from a PDF document along with their properties.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        dict: A dictionary where keys are page numbers and values are lists of
              dictionaries, each representing a form field.
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

    # Group fields by page
    fields_by_page = {}
    for field in fields:
        page_num = field['page']
        if page_num not in fields_by_page:
            fields_by_page[page_num] = []
        fields_by_page[page_num].append(field)
    
    doc.close() # Close the document after extraction
    return fields_by_page

def fill_pdf_form(pdf_bytes, field_mapping):
    """
    Fills a PDF form with provided data.

    Args:
        pdf_bytes (bytes): The content of the PDF file as bytes.
        field_mapping (dict): A dictionary where keys are field names and values
                              are dictionaries containing 'value' and 'type' for each field.

    Returns:
        bytes: The content of the filled PDF file as bytes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    filled_count = 0

    for page in doc:
        for widget in page.widgets() or []:
            field_name = widget.field_name
            if field_name in field_mapping:
                field_data = field_mapping[field_name]
                value = field_data.get("answer") # Use "answer" from the LLM output

                if value is not None and value != "Not documented" and value != "" and value != "Not applicable":
                    try:
                        if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                            # Convert "Yes"/"No" answers to boolean for checkboxes
                            if str(value).lower() in ['yes', 'true', '1']:
                                widget.field_value = True
                            else:
                                widget.field_value = False
                        else:
                            widget.field_value = str(value)
                        
                        widget.update()
                        filled_count += 1
                    except Exception as e:
                        print(f"Error filling field '{field_name}': {e}")
    
    print(f"Filled {filled_count} fields in the PDF.")
    
    # Save the filled PDF to a BytesIO object
    output_buffer = BytesIO()
    doc.save(output_buffer, deflate=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    output_buffer.seek(0)
    return output_buffer.getvalue()
