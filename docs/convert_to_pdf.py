"""
Convert PROJECT_REVIEW_GUIDE.md to a properly formatted PDF
"""

from fpdf import FPDF
import re
from pathlib import Path

class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        
    def header(self):
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'LungXAI Project Review Guide', 0, 0, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Arial', 'B', 18)
            self.set_text_color(0, 51, 102)
        elif level == 2:
            self.set_font('Arial', 'B', 14)
            self.set_text_color(0, 76, 153)
        else:
            self.set_font('Arial', 'B', 12)
            self.set_text_color(0, 102, 204)
        
        self.ln(5)
        self.multi_cell(0, 8, title)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font('Courier', '', 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        
        lines = text.split('\n')
        for line in lines:
            # Truncate very long lines
            if len(line) > 95:
                line = line[:92] + '...'
            self.cell(0, 5, line, 0, 1, 'L', fill=True)
        self.ln(3)

    def table_row(self, cells, is_header=False):
        self.set_font('Arial', 'B' if is_header else '', 9)
        if is_header:
            self.set_fill_color(0, 51, 102)
            self.set_text_color(255, 255, 255)
        else:
            self.set_fill_color(245, 245, 245)
            self.set_text_color(0, 0, 0)
        
        col_width = (self.w - 30) / len(cells)
        for cell in cells:
            # Truncate long cell content
            if len(cell) > 30:
                cell = cell[:27] + '...'
            self.cell(col_width, 7, cell, 1, 0, 'L', fill=True)
        self.ln()

    def bullet_point(self, text, indent=0):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.set_x(20 + indent * 10)
        self.multi_cell(0, 6, f"- {text}")

    def bold_text(self, text):
        self.set_font('Arial', 'B', 10)
        self.multi_cell(0, 6, text)
        self.set_font('Arial', '', 10)

def clean_text(text):
    """Clean markdown formatting from text"""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove backticks
    text = text.replace('`', '')
    # Remove $$ for math
    text = text.replace('$$', '')
    text = text.replace('$', '')
    # Convert special chars
    text = text.replace('→', '->')
    text = text.replace('←', '<-')
    text = text.replace('↓', 'v')
    text = text.replace('↑', '^')
    text = text.replace('✅', '[OK]')
    text = text.replace('✓', '[OK]')
    text = text.replace('❌', '[X]')
    text = text.replace('⚠', '[!]')
    text = text.replace('📁', '')
    text = text.replace('🎯', '')
    text = text.replace('🎓', '')
    text = text.replace('│', '|')
    text = text.replace('├', '|')
    text = text.replace('└', '+')
    text = text.replace('─', '-')
    text = text.replace('┌', '+')
    text = text.replace('┐', '+')
    text = text.replace('┘', '+')
    text = text.replace('┬', '+')
    text = text.replace('┴', '+')
    text = text.replace('┼', '+')
    text = text.replace('┤', '|')
    text = text.replace('├', '|')
    text = text.replace('═', '=')
    text = text.replace('║', '|')
    text = text.replace('╔', '+')
    text = text.replace('╗', '+')
    text = text.replace('╚', '+')
    text = text.replace('╝', '+')
    text = text.replace('╠', '+')
    text = text.replace('╣', '+')
    text = text.replace('╦', '+')
    text = text.replace('╩', '+')
    text = text.replace('╬', '+')
    text = text.replace('▼', 'v')
    text = text.replace('▲', '^')
    text = text.replace('◄', '<')
    text = text.replace('►', '>')
    text = text.replace('●', '*')
    text = text.replace('○', 'o')
    text = text.replace('■', '#')
    text = text.replace('□', '[ ]')
    text = text.replace('≥', '>=')
    text = text.replace('≤', '<=')
    text = text.replace('×', 'x')
    text = text.replace('÷', '/')
    text = text.replace('±', '+/-')
    text = text.replace('∑', 'Sum')
    text = text.replace('∂', 'd')
    text = text.replace('α', 'alpha')
    text = text.replace('β', 'beta')
    text = text.replace('γ', 'gamma')
    text = text.replace('°', ' deg')
    text = text.replace('ᶜ', 'c')
    text = text.replace('ₖ', 'k')
    text = text.replace('ᵢ', 'i')
    text = text.replace('ⱼ', 'j')
    text = text.replace('ᵏ', 'k')
    text = text.replace('Σ', 'Sum')
    text = text.replace('•', '-')
    
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', 'replace').decode('ascii')
    
    return text

def parse_markdown(md_content):
    """Parse markdown content into structured elements"""
    lines = md_content.split('\n')
    elements = []
    in_code_block = False
    code_buffer = []
    in_table = False
    table_buffer = []
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                elements.append(('code', '\n'.join(code_buffer)))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            continue
        
        if in_code_block:
            code_buffer.append(clean_text(line))
            continue
        
        # Handle tables
        if '|' in line and not line.strip().startswith('```'):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells and not all(c.replace('-', '').replace(':', '') == '' for c in cells):
                if not in_table:
                    in_table = True
                    table_buffer = []
                table_buffer.append(cells)
            continue
        elif in_table:
            if table_buffer:
                elements.append(('table', table_buffer))
            table_buffer = []
            in_table = False
        
        # Handle headers
        if line.startswith('# '):
            elements.append(('h1', clean_text(line[2:].strip())))
        elif line.startswith('## '):
            elements.append(('h2', clean_text(line[3:].strip())))
        elif line.startswith('### '):
            elements.append(('h3', clean_text(line[4:].strip())))
        elif line.startswith('#### '):
            elements.append(('h3', clean_text(line[5:].strip())))
        # Handle bullets
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            indent = len(line) - len(line.lstrip())
            text = line.strip()[2:].strip()
            elements.append(('bullet', clean_text(text), indent // 2))
        elif re.match(r'^\d+\.\s', line.strip()):
            text = re.sub(r'^\d+\.\s', '', line.strip())
            elements.append(('bullet', clean_text(text), 0))
        # Handle blockquotes
        elif line.strip().startswith('>'):
            elements.append(('quote', clean_text(line.strip()[1:].strip())))
        # Handle horizontal rules
        elif line.strip() in ['---', '***', '___']:
            elements.append(('hr', ''))
        # Handle regular text
        elif line.strip():
            elements.append(('text', clean_text(line.strip())))
        else:
            elements.append(('blank', ''))
    
    # Handle any remaining table
    if in_table and table_buffer:
        elements.append(('table', table_buffer))
    
    return elements

def create_pdf(md_path, pdf_path):
    """Create PDF from markdown file"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    elements = parse_markdown(content)
    
    pdf = MarkdownPDF()
    
    for elem in elements:
        elem_type = elem[0]
        
        try:
            if elem_type == 'h1':
                pdf.add_page()
                pdf.chapter_title(elem[1], 1)
            elif elem_type == 'h2':
                pdf.chapter_title(elem[1], 2)
            elif elem_type == 'h3':
                pdf.chapter_title(elem[1], 3)
            elif elem_type == 'text':
                pdf.body_text(elem[1])
            elif elem_type == 'bullet':
                indent = elem[2] if len(elem) > 2 else 0
                pdf.bullet_point(elem[1], indent)
            elif elem_type == 'code':
                pdf.code_block(elem[1])
            elif elem_type == 'table':
                rows = elem[1]
                if rows:
                    # First row is header
                    pdf.table_row([clean_text(c) for c in rows[0]], is_header=True)
                    for row in rows[1:]:
                        pdf.table_row([clean_text(c) for c in row], is_header=False)
                    pdf.ln(3)
            elif elem_type == 'quote':
                pdf.set_font('Arial', 'I', 10)
                pdf.set_text_color(80, 80, 80)
                pdf.set_x(25)
                pdf.multi_cell(0, 6, elem[1])
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            elif elem_type == 'hr':
                pdf.ln(3)
                pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
                pdf.ln(5)
            elif elem_type == 'blank':
                pdf.ln(2)
        except Exception as e:
            print(f"Warning: Could not process element: {elem_type} - {str(e)[:50]}")
            continue
    
    pdf.output(pdf_path)
    print(f"PDF created successfully: {pdf_path}")

if __name__ == "__main__":
    docs_dir = Path(__file__).parent
    md_file = docs_dir / "PROJECT_REVIEW_GUIDE.md"
    pdf_file = docs_dir / "PROJECT_REVIEW_GUIDE.pdf"
    
    create_pdf(md_file, pdf_file)
