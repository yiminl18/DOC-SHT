import pdfplumber
import json
import os
from typing import List, Dict, Any, Tuple
import re


def extract_phrases_with_pdfplumber(pdf_path: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    Extract words from a PDF using pdfplumber and save to local file.
    Based on the get_patterns approach - extracts individual words with their patterns.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the extracted words (optional)
    
    Returns:
        List of dictionaries containing word information with visual patterns
    """
    if output_file is None:
        # Generate output filename based on input PDF name
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = f"out/{base_name}_pdfplumber_words.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    words_data = []
    word_id = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            
            for page_num, page in enumerate(pdf.pages):
                print(f"Processing page {page_num + 1}...")
                
                # Extract words with text flow and extra attributes
                words = page.extract_words(
                    use_text_flow=True, 
                    extra_attrs=["fontname", "size"]
                )
                
                for word in words:
                    # Skip empty words
                    if not word['text'].strip():
                        continue
                    
                    # Create bounding box
                    bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                    
                    # Get font information
                    font_name = word.get('fontname', 'Unknown')
                    font_size = round(word.get('size', 12), 3)
                    
                    # Calculate visual pattern features
                    is_bold = _is_bold_font(font_name)
                    is_all_cap = _is_all_caps(word['text'])
                    starts_with_number = _starts_with_number(word['text'])
                    is_center = _is_centered(bbox, page.width)
                    is_underline = _has_underline_word(word)
                    
                    # Create word dictionary
                    word_dict = {
                        'id': word_id,
                        'phrase': word['text'],  # Keep 'phrase' key for compatibility
                        'bbox': bbox,
                        'page': page_num,
                        'font': font_name,
                        'size': font_size,
                        'bold': 1 if is_bold else 0,
                        'all_cap': 1 if is_all_cap else 0,
                        'num_st': 1 if starts_with_number else 0,
                        'is_center': 1 if is_center else 0,
                        'is_underline': 1 if is_underline else 0
                    }
                    
                    words_data.append(word_dict)
                    word_id += 1
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
    
    # Save to file (preserving pdfplumber's natural reading order)
    # try:
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(words_data, f, indent=2, ensure_ascii=False)
    #     print(f"Extracted {len(words_data)} words and saved to {output_file}")
    # except Exception as e:
    #     print(f"Error saving to file: {e}")
    
    return words_data


def _is_same_line(word1: Dict, word2: Dict, tolerance: float = 5.0) -> bool:
    """
    Check if two words are on the same line based on y-coordinates.
    
    Args:
        word1: First word dictionary
        word2: Second word dictionary
        tolerance: Y-coordinate tolerance in points
    
    Returns:
        True if words are on the same line
    """
    y1 = word1['top']
    y2 = word2['top']
    return abs(y1 - y2) <= tolerance


def _get_phrase_bbox(words: List[Dict]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box for a list of words.
    
    Args:
        words: List of word dictionaries
    
    Returns:
        Tuple of (x0, y0, x1, y1) coordinates
    """
    if not words:
        return (0, 0, 0, 0)
    
    x0 = min(word['x0'] for word in words)
    y0 = min(word['top'] for word in words)
    x1 = max(word['x1'] for word in words)
    y1 = max(word['bottom'] for word in words)
    
    return (x0, y0, x1, y1)


def _create_phrase_dict(words: List[Dict], page_num: int, phrase_id: int, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """
    Create a phrase dictionary with all visual pattern features.
    
    Args:
        words: List of word dictionaries
        page_num: Page number (0-indexed)
        phrase_id: Unique phrase ID
        bbox: Bounding box coordinates
    
    Returns:
        Dictionary with phrase information and visual patterns
    """
    # Combine text from all words
    phrase_text = ' '.join(word['text'] for word in words)
    
    # Get font properties (use the most common font in the phrase)
    font_counts = {}
    size_counts = {}
    
    for word in words:
        font = word.get('fontname', 'Unknown')
        size = word.get('size', 12)
        font_counts[font] = font_counts.get(font, 0) + 1
        size_counts[size] = size_counts.get(size, 0) + 1
    
    # Use most common font and size
    font_name = max(font_counts.items(), key=lambda x: x[1])[0]
    font_size = max(size_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate visual pattern features
    is_bold = _is_bold_font(font_name)
    is_all_cap = _is_all_caps(phrase_text)
    starts_with_number = _starts_with_number(phrase_text)
    is_center = _is_centered(bbox, words[0].get('page_width', 612))  # Default page width
    is_underline = _has_underline(words)
    
    return {
        'id': phrase_id,
        'phrase': phrase_text,
        'bbox': bbox,
        'page': page_num,
        'font': font_name,
        'size': font_size,
        'bold': 1 if is_bold else 0,
        'all_cap': 1 if is_all_cap else 0,
        'num_st': 1 if starts_with_number else 0,
        'is_center': 1 if is_center else 0,
        'is_underline': 1 if is_underline else 0
    }


def _is_bold_font(font_name: str) -> bool:
    """
    Check if font is bold based on font name.
    
    Args:
        font_name: Font name string
    
    Returns:
        True if font appears to be bold
    """
    font_lower = font_name.lower()
    bold_indicators = ['bold', 'b', 'heavy', 'black', 'demibold']
    return any(indicator in font_lower for indicator in bold_indicators)


def _is_all_caps(text: str) -> bool:
    """
    Check if text is all uppercase.
    
    Args:
        text: Text string
    
    Returns:
        True if text is all uppercase
    """
    if not text:
        return False
    return text.isupper() and text.isalpha()


def _starts_with_number(text: str) -> bool:
    """
    Check if text starts with a number.
    
    Args:
        text: Text string
    
    Returns:
        True if text starts with a number
    """
    if not text:
        return False
    return text[0].isdigit()


def _is_centered(bbox: Tuple[float, float, float, float], page_width: float, tolerance: float = 0.1) -> bool:
    """
    Check if phrase is centered on the page.
    
    Args:
        bbox: Bounding box coordinates
        page_width: Width of the page
        tolerance: Tolerance for centering (fraction of page width)
    
    Returns:
        True if phrase is centered
    """
    phrase_center = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    center_tolerance = page_width * tolerance
    
    return abs(phrase_center - page_center) <= center_tolerance


def _has_underline_word(word: Dict) -> bool:
    """
    Check if a word has underline.
    Note: pdfplumber doesn't always detect underlines reliably.
    
    Args:
        word: Word dictionary from pdfplumber
    
    Returns:
        True if underline is detected
    """
    # pdfplumber doesn't always provide underline information
    # This is a placeholder - in practice, you might need to analyze
    # the raw PDF objects or use additional heuristics
    return False


def _has_underline(words: List[Dict]) -> bool:
    """
    Check if any word in the phrase has underline.
    Note: pdfplumber doesn't always detect underlines reliably.
    
    Args:
        words: List of word dictionaries
    
    Returns:
        True if underline is detected
    """
    # pdfplumber doesn't always provide underline information
    # This is a placeholder - in practice, you might need to analyze
    # the raw PDF objects or use additional heuristics
    return False


def extract_words_simple(pdf_path: str, output_file: str = None) -> List[str]:
    """
    Extract only words from a PDF using pdfplumber and save to local file.
    Returns a simple list of words without any visual features.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the extracted words (optional)
    
    Returns:
        List of words as strings
    """
    if output_file is None:
        # Generate output filename based on input PDF name
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = f"out/{base_name}_simple_words.txt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    words = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            
            for page_num, page in enumerate(pdf.pages):
                print(f"Processing page {page_num + 1}...")
                
                # Extract words with text flow
                page_words = page.extract_words(use_text_flow=True)
                
                for word in page_words:
                    text = word['text'].strip()
                    if text:  # Only add non-empty words
                        words.append(text)
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + '\n')
        print(f"Extracted {len(words)} words and saved to {output_file}")
    except Exception as e:
        print(f"Error saving to file: {e}")
    
    return words


def extract_words_json_simple(pdf_path: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    Extract words from a PDF using pdfplumber and save to local file as JSON.
    Returns minimal word information (text, page, position) without visual features.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the extracted words (optional)
    
    Returns:
        List of dictionaries with minimal word information
    """
    if output_file is None:
        # Generate output filename based on input PDF name
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = f"out/{base_name}_simple_words.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    words_data = []
    word_id = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            
            for page_num, page in enumerate(pdf.pages):
                print(f"Processing page {page_num + 1}...")
                
                # Extract words with text flow
                page_words = page.extract_words(use_text_flow=True)
                
                for word in page_words:
                    text = word['text'].strip()
                    if text:  # Only add non-empty words
                        # Create minimal word dictionary
                        word_dict = {
                            'id': word_id,
                            'text': text,
                            'page': page_num,
                            'bbox': (word['x0'], word['top'], word['x1'], word['bottom'])
                        }
                        
                        words_data.append(word_dict)
                        word_id += 1
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
    
    # Save to file (preserving pdfplumber's natural reading order)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(words_data, f, indent=2, ensure_ascii=False)
        print(f"Extracted {len(words_data)} words and saved to {output_file}")
    except Exception as e:
        print(f"Error saving to file: {e}")
    
    return words_data


def load_phrases_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load phrases from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        List of phrase dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading phrases from file: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Example usage
    pdf_file = "raw_data/paper/A Lived Informatics Model of Personal Informatics.pdf"
    
    # Extract words with all features
    print("=== Extracting words with all features ===")
    words_with_features = extract_phrases_with_pdfplumber(pdf_file, "out/paper_pdfplumber_words.json")
    print(f"Extracted {len(words_with_features)} words with features")
    
    
