#!/usr/bin/env python3
"""
PDF Debug Analyzer - Python version of Debug.ps1
Extracts text with coordinates and tests footer filtering
"""

import pdfplumber
import sys
from collections import defaultdict

def clean_footer_from_cell(cell_text, column_name):
    """Retrospectively clean footer components from cell text"""
    import re
    cleaned = cell_text

    # User column: Remove "ion Date : YYYY-MM-DD" or variants
    if column_name == "User":
        cleaned = re.sub(r'ion Date\s*:\s*\d{4}-\d{2}-\d{2}', '', cleaned)
        cleaned = re.sub(r'Date\s*:\s*\d{4}-\d{2}-\d{2}', '', cleaned)

    # Group column: Remove timestamp fragments (HH:MM or HH:MM:SS)
    if column_name == "Group":
        cleaned = re.sub(r'\s*\d{1,2}:\d{2}(?::\d{2})?\s*$', '', cleaned)

    # Time column: Remove everything AFTER first GMT±HH:MM
    if "Time" in column_name:
        match = re.search(r'(GMT[+-]\d{2}:\d{2})', cleaned)
        if match:
            cleaned = cleaned[:match.end()]

    # Error column: Remove "Page X/Y" (with or without space)
    if "Error" in column_name:
        cleaned = re.sub(r'Page\s*\d+/\d+', '', cleaned)

    return cleaned.strip()

def is_footer_chunk(text):
    """Tests if a text chunk is footer content - IMPROVED patterns"""
    import re
    text = text.strip()

    # Pattern 1: "Page X/Y" - mit oder OHNE Leerzeichen!
    if re.match(r'^Page\s*\d+/\d+$', text):
        return True

    # Pattern 1b: Nur "Page" (separater Chunk)
    if re.match(r'^Page$', text):
        return True

    # Pattern 1c: Nur Seitenzahl "X/Y" oder "X/YYYY" (separater Chunk)
    if re.match(r'^\d+/\d+$', text):
        return True

    # Pattern 2: Text contains "Date" AND date follows
    if re.search(r'Date.*\d{4}-\d{2}-\d{2}', text):
        return True

    # Pattern 2b: Text ends with "Date" or "Date:"
    if re.search(r'Date:?\s*$', text):
        return True

    # Pattern 3: Standalone date (mit ODER ohne Prefix!)
    # Matcht: ": 2024-09-16", "2024-09-16", "- 2024-09-16"
    if re.match(r'^[:;\-]?\s*\d{4}-\d{2}-\d{2}$', text):
        return True

    # Pattern 4a: Zeitstempel mit trailing colon OR Sekunden
    # Matcht: "12:28:", "12:28:10"
    if re.search(r'\d{1,2}:\d{2}:(?:\d{2})?\s*$', text):
        return True

    # Pattern 5a: GMT timezone (mit ODER ohne Zahlen-Prefix!)
    # Matcht: "10 GMT+02:00", "GMT+02:00"
    if re.match(r'^(?:\d{1,2}\s+)?GMT[+-]\d{2}:\d{2}$', text):
        return True

    # Pattern 6: Number + GMT prefix
    if re.match(r'^\d{1,2}\s+GMT', text):
        return True

    return False

def extract_pdf_text_with_coords(pdf_path):
    """Extract all text with X/Y coordinates from PDF"""
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract words with their positions
            words = page.extract_words()
            for word in words:
                chunks.append({
                    'page': page_num,
                    'x': round(word['x0'], 2),
                    'y': round(page.height - word['top'], 2),  # Convert to bottom-up coordinates
                    'text': word['text']
                })

    return chunks

def group_chunks_by_y(chunks, tolerance=0.5):
    """Group chunks by Y position (physical lines)"""
    y_groups = defaultdict(list)

    for chunk in chunks:
        y = chunk['y']
        found_group = False

        # Find existing Y group within tolerance
        for existing_y in list(y_groups.keys()):
            if abs(y - existing_y) < tolerance:
                y_groups[existing_y].append(chunk)
                found_group = True
                break

        if not found_group:
            y_groups[y] = [chunk]

    return y_groups

def analyze_pdf(pdf_path):
    """Main analysis function"""
    print("=" * 60)
    print(" PDF ANALYSIS - Python Debug Tool")
    print("=" * 60)
    print(f"Analyzing: {pdf_path}\n")

    # Extract chunks
    chunks = extract_pdf_text_with_coords(pdf_path)
    print(f"Total chunks extracted: {len(chunks)}\n")

    # Group by Y
    y_groups = group_chunks_by_y(chunks)
    print(f"Y-Groups: {len(y_groups)}\n")

    # Show first 10 Y-groups
    print("=== FIRST 10 Y-GROUPS ===\n")
    sorted_y = sorted(y_groups.keys(), reverse=True)[:10]
    for y in sorted_y:
        print(f"Y={y:.2f}")
        for chunk in sorted(y_groups[y], key=lambda c: c['x']):
            print(f"  X={chunk['x']:7.2f} | '{chunk['text']}'")
        print()

    # Find header
    print("=== HEADER DETECTION ===\n")
    header_y = None
    for y in sorted(y_groups.keys(), reverse=True):
        y_chunks = y_groups[y]
        texts = [c['text'] for c in y_chunks]
        combined_text = ' '.join(texts)

        if 'User' in texts and 'Group' in texts and 'Category' in texts:
            header_y = y
            print(f"Header found at Y={header_y:.2f}")
            print(f"Header text: {combined_text}")
            break

    if not header_y:
        print("ERROR: No header found!")
        return

    # Define columns
    print("\n=== COLUMN DEFINITION ===\n")
    header_chunks = [c for c in chunks if abs(c['y'] - header_y) < 2.0 and
                     any(keyword in c['text'] for keyword in ['User', 'Group', 'Time', 'Succeeded', 'Category', 'Reason', 'Action', 'Error'])]
    header_chunks = sorted(header_chunks, key=lambda c: c['x'])

    columns = []
    for i, hc in enumerate(header_chunks):
        x_end = header_chunks[i+1]['x'] if i+1 < len(header_chunks) else 1000
        columns.append({
            'name': hc['text'],
            'x_start': hc['x'],
            'x_end': x_end
        })
        print(f"{hc['text']:15} X={hc['x']:.2f} to X={x_end:.2f}")

    # Find footer
    print("\n=== FOOTER DETECTION ===\n")
    footer_y = None
    for y in y_groups.keys():
        y_chunks = y_groups[y]
        texts = [c['text'] for c in y_chunks]
        combined_text = ' '.join(texts)

        if 'Creation' in combined_text and 'Page' in combined_text:
            footer_y = y
            print(f"Footer line found at Y={footer_y:.2f}")
            print(f"Footer text: {combined_text}")
            break

    if not footer_y:
        print("WARNING: No separate footer line found - footer chunks might be inline")

    # Find table rows
    print("\n=== TABLE ROW DETECTION ===\n")
    user_col = columns[0]
    group_col = columns[1]
    table_row_starts = []

    for y in sorted(y_groups.keys(), reverse=True):
        if y >= header_y - 5:  # Skip header area
            continue

        y_chunks = y_groups[y]

        # Check if User and Group columns are filled
        user_chunks = [c for c in y_chunks if user_col['x_start'] <= c['x'] < user_col['x_end']]
        group_chunks = [c for c in y_chunks if group_col['x_start'] <= c['x'] < group_col['x_end']]
        group_text = ''.join([c['text'] for c in sorted(group_chunks, key=lambda c: c['x'])])

        if user_chunks and 'ICPMH' in group_text:
            user_text = ' '.join([c['text'] for c in sorted(user_chunks, key=lambda c: c['x'])])
            table_row_starts.append(y)
            print(f"Row found at Y={y:.2f} | User: '{user_text}' | Group: '{group_text}'")

    print(f"\nTotal table rows found: {len(table_row_starts)}")

    # Analyze LAST table row in detail
    if table_row_starts:
        print("\n" + "=" * 60)
        print("=== LAST TABLE ROW ANALYSIS (Footer Issues) ===")
        print("=" * 60 + "\n")

        table_row_starts = sorted(table_row_starts, reverse=True)
        last_row_y = table_row_starts[-1]
        row_end_y = -100  # No row below

        print(f"Last row Y: {last_row_y:.2f}\n")

        # Get all chunks in this row
        row_chunks = [c for c in chunks if c['y'] <= last_row_y and c['y'] > row_end_y]

        print(f"Total chunks in last row: {len(row_chunks)}\n")
        print(f"Row Y range: {last_row_y:.2f} to {row_end_y:.2f}")
        footer_y_str = f"{footer_y:.2f}" if footer_y else "None"
        print(f"Footer Y (from detection): {footer_y_str}\n")

        # Test footer filtering: Chunks near footer_y (within 10 pixels) AND matching pattern
        print("--- TESTING FOOTER FILTERING (Y-based + Pattern) ---\n")
        footer_chunks_found = []
        valid_chunks = []

        for chunk in row_chunks:
            is_near_footer = footer_y and abs(chunk['y'] - footer_y) < 10.0
            matches_pattern = is_footer_chunk(chunk['text'])

            if is_near_footer and matches_pattern:
                footer_chunks_found.append(chunk)
                print(f"[FOOTER] X={chunk['x']:7.2f} Y={chunk['y']:7.2f} | '{chunk['text']}'")
            else:
                valid_chunks.append(chunk)

        print(f"\nFooter chunks detected: {len(footer_chunks_found)}")
        print(f"Valid chunks remaining: {len(valid_chunks)}\n")

        # Debug: Show Error column chunks
        print("--- DEBUG: Chunks in Error column ---\n")
        error_col = columns[-1]  # Last column is Error
        error_chunks = [c for c in row_chunks if error_col['x_start'] <= c['x'] < error_col['x_end']]
        for chunk in error_chunks:
            is_near_footer = footer_y and abs(chunk['y'] - footer_y) < 10.0
            matches_pattern = is_footer_chunk(chunk['text'])
            print(f"X={chunk['x']:7.2f} Y={chunk['y']:7.2f} | '{chunk['text']}' | Near footer: {is_near_footer} | Matches pattern: {matches_pattern}")
        print()

        # Extract cells WITH retrospective cleaning
        print("--- CELL CONTENTS (WITH Retrospective String Cleaning) ---\n")

        for col in columns[:8]:  # First 8 columns
            col_chunks = [c for c in row_chunks if col['x_start'] <= c['x'] < col['x_end']]
            col_chunks = sorted(col_chunks, key=lambda c: (-c['y'], c['x']))

            # Combine text
            cell_text = ''
            last_y = None
            for chunk in col_chunks:
                if last_y is not None and abs(chunk['y'] - last_y) > 1.0:
                    cell_text += ' '
                cell_text += chunk['text']
                last_y = chunk['y']

            # Apply retrospective cleaning
            cleaned_text = clean_footer_from_cell(cell_text.strip(), col['name'])

            print(f"{col['name']:15} : '{cleaned_text}'")

        print("\n" + "=" * 60)

        # Show what WOULD be in cells WITHOUT filtering
        print("\n--- CELL CONTENTS (WITHOUT Footer Filtering) ---\n")

        for col in columns[:8]:
            col_chunks = [c for c in row_chunks if col['x_start'] <= c['x'] < col['x_end']]
            col_chunks = sorted(col_chunks, key=lambda c: (-c['y'], c['x']))

            cell_text = ''
            last_y = None
            for chunk in col_chunks:
                if last_y is not None and abs(chunk['y'] - last_y) > 1.0:
                    cell_text += ' '
                cell_text += chunk['text']
                last_y = chunk['y']

            print(f"{col['name']:15} : '{cell_text.strip()}'")

if __name__ == '__main__':
    pdf_path = 'sample mh.pdf'
    try:
        analyze_pdf(pdf_path)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
