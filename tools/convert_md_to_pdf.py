import markdown
from xhtml2pdf import pisa
import sys
import os

def convert_md_to_pdf(source_md, output_pdf):
    print(f"Reading {source_md}...")
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # CSS for basic styling
    css = """
    <style>
        @page { size: A4; margin: 2cm; }
        body { font-family: Helvetica, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }
        h1 { color: #2c3e50; font-size: 24pt; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; }
        h2 { color: #34495e; font-size: 18pt; margin-top: 25px; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        h3 { color: #7f8c8d; font-size: 14pt; margin-top: 20px; margin-bottom: 8px; }
        ul { margin-bottom: 10px; }
        li { margin-bottom: 5px; }
        p { margin-bottom: 10px; }
        code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: "Courier New", monospace; font-size: 10pt; color: #e74c3c; }
        pre { background-color: #f8f9fa; padding: 10px; border: 1px solid #e1e4e8; border-radius: 4px; overflow-x: auto; margin-bottom: 15px; }
        pre code { background-color: transparent; padding: 0; color: #333; border: none; }
        blockquote { border-left: 4px solid #3498db; padding-left: 15px; color: #555; font-style: italic; background-color: #f0f7fb; padding: 10px; margin: 10px 0; }
        strong { color: #000; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
    </style>
    """
    
    print("Converting Markdown to HTML...")
    # Add extra extension for better parsing
    html_content = markdown.markdown(text, extensions=['fenced_code', 'tables', 'sane_lists'])
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {css}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    print(f"Generating PDF at {output_pdf}...")
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file)

    if pisa_status.err:
        print(f"Error: {pisa_status.err}")
        sys.exit(1)
    else:
        print(f"Successfully created PDF: {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_md_to_pdf.py <input_md> <output_pdf>")
        sys.exit(1)
    
    convert_md_to_pdf(sys.argv[1], sys.argv[2])
