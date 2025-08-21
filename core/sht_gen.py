import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
import os
import random

def is_valid_phrase(text: str) -> bool:
    """
    Check if a text phrase is valid (not empty, not just whitespace, not just special characters).
    
    Args:
        text (str): Text to validate
        
    Returns:
        bool: True if text is valid, False otherwise
    """
    if not text:
        return False
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if empty after stripping
    if not text:
        return False
    
    # Check if only whitespace
    if text.isspace():
        return False
    
    # Check if only special/invisible characters (common in PDFs)
    # Remove common invisible characters and check if anything meaningful remains
    cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', cleaned_text)  # Zero-width spaces
    
    if not cleaned_text.strip():
        return False
    
    # Check if the cleaned text has at least one printable character
    if not any(c.isprintable() for c in cleaned_text):
        return False
    
    return True 


def phrase_visual_pattern_extraction(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract phrases and their visual patterns from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        List[Dict[str, Any]]: List of phrase objects with visual properties
        Each object contains:
        - phrase: extracted text
        - bbox: bounding box (x0, y0, x1, y1)
        - page: page number (0-indexed)
        - font: font type
        - size: font size
        - bold: 0 or 1 indicating if text is bold
        - is_underline: 0 or 1 indicating if text is underlined
        - all_cap: 0 or 1 indicating if all letters are capitalized
        - num_st: 0 or 1 indicating if phrase starts with a number
        - is_center: 0 or 1 indicating if phrase is centered on the page
    """
    
    def is_scanned_pdf(doc) -> bool:
        """Determine if PDF is scanned (image-based) or text-based."""
        first_page = doc[0]
        text = first_page.get_text()
        # If very little text is extracted, likely scanned
        return len(text.strip()) < 100
    
    def extract_text_from_scanned_pdf(doc) -> List[Dict[str, Any]]:
        """Extract text from scanned PDF using OCR."""
        phrases = []
        phrase_id = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page dimensions
            rect = page.rect
            width, height = rect.width, rect.height
            
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # Scale factor for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Use OCR to extract text with bounding boxes
            try:
                # Use pytesseract with detailed output
                ocr_data = pytesseract.image_to_data(
                    img, 
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Assume uniform block of text
                )
                
                # Process OCR results
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    # Filter out empty phrases, whitespace-only phrases, and special characters
                    if (is_valid_phrase(text) and int(ocr_data['conf'][i]) > 30):  # Confidence threshold
                        # Get bounding box (normalize to page coordinates)
                        x0 = ocr_data['left'][i] / 2  # Divide by scale factor
                        y0 = ocr_data['top'][i] / 2
                        x1 = x0 + (ocr_data['width'][i] / 2)
                        y1 = y0 + (ocr_data['height'][i] / 2)
                        
                        # Estimate font properties from OCR
                        font_size = ocr_data['height'][i] / 2
                        
                        # Simple heuristic for bold detection
                        # Higher confidence and larger height might indicate bold
                        is_bold = 1 if (int(ocr_data['conf'][i]) > 70 and font_size > 20) else 0
                        
                        # Simple heuristic for underline detection
                        # For OCR, we can't reliably detect underline, so set to 0
                        # In practice, underlined text might be detected as separate elements
                        is_underline = 0
                        
                        # Check if all letters are capitalized
                        all_cap = 1 if text.isupper() and text.isalpha() else 0
                        
                        # Check if phrase starts with a number
                        num_st = 1 if text and text[0].isdigit() else 0
                        
                        # Check if phrase is centered on the page
                        page_width = width
                        phrase_center_x = (x0 + x1) / 2
                        page_center_x = page_width / 2
                        # Heuristic: consider centered if within 20% of page center
                        center_threshold = page_width * 0.2
                        is_center = 1 if abs(phrase_center_x - page_center_x) <= center_threshold else 0
                        
                        phrases.append({
                            'id': phrase_id,
                            'phrase': text,
                            'bbox': (x0, y0, x1, y1),
                            'page': page_num,
                            'font': 'OCR_Detected',
                            'size': font_size,
                            'bold': is_bold,
                            'is_underline': is_underline,
                            'all_cap': all_cap,
                            'num_st': num_st,
                            'is_center': is_center
                        })
                        phrase_id += 1
                        
            except Exception as e:
                print(f"OCR failed for page {page_num}: {e}")
                continue
                
        return phrases
    
    def extract_text_from_normal_pdf(doc) -> List[Dict[str, Any]]:
        """Extract text from normal (text-based) PDF."""
        phrases = []
        phrase_id = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with detailed information
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            # Filter out empty phrases, whitespace-only phrases, and special characters
                            if is_valid_phrase(text):
                                # Get bounding box
                                bbox = span["bbox"]
                                
                                # Get font information
                                font_name = span.get("font", "Unknown")
                                font_size = span.get("size", 12)
                                
                                # Determine if text is bold
                                # Check font flags or font name for bold indicators
                                is_bold = 0
                                if "bold" in font_name.lower() or "Bold" in font_name:
                                    is_bold = 1
                                elif "flags" in span:
                                    # Font flags: bit 20 is bold
                                    flags = span["flags"]
                                    if flags & (1 << 20):  # Bold flag
                                        is_bold = 1
                                
                                # Determine if text is underlined
                                # Check font flags for underline indicators
                                is_underline = 0
                                if "flags" in span:
                                    flags = span["flags"]
                                    if flags & (1 << 4):  # Underline flag (bit 4)
                                        is_underline = 1
                                
                                # Check if all letters are capitalized
                                all_cap = 1 if text.isupper() and text.isalpha() else 0
                                
                                # Check if phrase starts with a number
                                num_st = 1 if text and text[0].isdigit() else 0
                                
                                # Check if phrase is centered on the page
                                page_width = page.rect.width
                                phrase_center_x = (bbox[0] + bbox[2]) / 2
                                page_center_x = page_width / 2
                                # Heuristic: consider centered if within 20% of page center
                                center_threshold = page_width * 0.2
                                is_center = 1 if abs(phrase_center_x - page_center_x) <= center_threshold else 0
                                
                                # Each span already has consistent visual properties (font, size, bold)
                                # No need to split - treat each span as a phrase
                                phrases.append({
                                    'id': phrase_id,
                                    'phrase': text,
                                    'bbox': bbox,
                                    'page': page_num,
                                    'font': font_name,
                                    'size': font_size,
                                    'bold': is_bold,
                                    'is_underline': is_underline,
                                    'all_cap': all_cap,
                                    'num_st': num_st,
                                    'is_center': is_center
                                })
                                phrase_id += 1
        
        return phrases
    
    def sort_phrases_by_reading_order(phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort phrases by reading order (top to bottom, left to right)."""
        def sort_key(phrase):
            bbox = phrase['bbox']
            page = phrase['page']
            # Primary sort by page, then by y-coordinate, then by x-coordinate
            return (page, bbox[1], bbox[0])
        
        return sorted(phrases, key=sort_key)
    
    try:
        # Open the PDF document
        doc = fitz.open(file_path)
        
        if not doc:
            raise ValueError("Could not open PDF file")
        
        # Determine if PDF is scanned or normal
        scanned = is_scanned_pdf(doc)
        
        if scanned:
            print("Detected scanned PDF - using OCR")
            phrases = extract_text_from_scanned_pdf(doc)
        else:
            print("Detected normal PDF - extracting text directly")
            phrases = extract_text_from_normal_pdf(doc)
        
        # Sort phrases by reading order
        phrases = sort_phrases_by_reading_order(phrases)
        
        # Clean up
        doc.close()
        
        return phrases
        
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []


def save_phrases_to_json(phrases: List[Dict[str, Any]], output_file: str):
    """Save extracted phrases to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(phrases, f, indent=2, ensure_ascii=False)


def phrase_merge(phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive phrases with the same visual properties.
    
    Args:
        phrases (List[Dict[str, Any]]): List of phrase objects with visual properties
        
    Returns:
        List[Dict[str, Any]]: List of merged phrase objects
    """
    if not phrases:
        return []
    
    merged_phrases = []
    current_group = [phrases[0]]
    merged_id = 0
    
    for i in range(1, len(phrases)):
        current_phrase = phrases[i]
        last_phrase = current_group[-1]
        
        # Check if current phrase can be merged with the last phrase in current group
        can_merge = (
            current_phrase['page'] == last_phrase['page'] and
            current_phrase['font'] == last_phrase['font'] and
            current_phrase['size'] == last_phrase['size'] and
            current_phrase['bold'] == last_phrase['bold'] and
            current_phrase['is_underline'] == last_phrase['is_underline'] and
            current_phrase['all_cap'] == last_phrase['all_cap']
        )
        
        # Additional check: don't merge underlined and bold phrases if they're not in the same row
        if can_merge and (current_phrase['is_underline'] == 1 and current_phrase['bold'] == 1):
            # Check if phrases are in the same row (similar y-coordinates)
            current_y_center = (current_phrase['bbox'][1] + current_phrase['bbox'][3]) / 2
            last_y_center = (last_phrase['bbox'][1] + last_phrase['bbox'][3]) / 2
            
            # Use font size as tolerance for row detection
            font_size = current_phrase['size']
            y_tolerance = font_size * 0.5  # Allow half a font size difference
            
            # If y-coordinates differ significantly, don't merge
            if abs(current_y_center - last_y_center) > y_tolerance:
                can_merge = False
        
        if can_merge:
            # Add to current group for merging
            current_group.append(current_phrase)
        else:
            # Merge current group and add to result
            merged_phrase = merge_phrase_group(current_group, merged_id)
            merged_phrases.append(merged_phrase)
            merged_id += 1
            
            # Start new group with current phrase
            current_group = [current_phrase]
    
    # Don't forget to merge the last group
    if current_group:
        merged_phrase = merge_phrase_group(current_group, merged_id)
        merged_phrases.append(merged_phrase)
    
    return merged_phrases


def merge_phrase_group(phrase_group: List[Dict[str, Any]], merged_id: int) -> Dict[str, Any]:
    """
    Merge a group of phrases with the same visual properties into a single phrase.
    
    Args:
        phrase_group (List[Dict[str, Any]]): List of phrases to merge
        merged_id (int): ID to assign to the merged phrase
        
    Returns:
        Dict[str, Any]: Merged phrase object
    """
    if len(phrase_group) == 1:
        # Update the ID for single phrases
        result = phrase_group[0].copy()
        result['id'] = merged_id
        return result
    
    # Merge text
    merged_text = " ".join([phrase['phrase'] for phrase in phrase_group])
    
    # Merge bounding boxes
    x0 = min(phrase['bbox'][0] for phrase in phrase_group)
    y0 = min(phrase['bbox'][1] for phrase in phrase_group)
    x1 = max(phrase['bbox'][2] for phrase in phrase_group)
    y1 = max(phrase['bbox'][3] for phrase in phrase_group)
    merged_bbox = (x0, y0, x1, y1)
    
    # Use properties from the first phrase (they should all be the same)
    first_phrase = phrase_group[0]
    
    # Determine if merged phrase is centered
    # Calculate center of merged bbox
    merged_center_x = (x0 + x1) / 2
    # We need page width to determine centering - use from first phrase
    # This is approximate since we don't have page width here
    # For now, use the original is_center value from first phrase
    merged_is_center = first_phrase['is_center']
    
    # Determine if merged phrase starts with number
    merged_num_st = 1 if merged_text and merged_text[0].isdigit() else 0
    
    # Determine if merged phrase is all caps
    merged_all_cap = 1 if merged_text.isupper() and merged_text.isalpha() else 0
    
    return {
        'id': merged_id,
        'phrase': merged_text,
        'bbox': merged_bbox,
        'page': first_phrase['page'],
        'font': first_phrase['font'],
        'size': first_phrase['size'],
        'bold': first_phrase['bold'],
        'is_underline': first_phrase['is_underline'],
        'all_cap': merged_all_cap,
        'num_st': merged_num_st,
        'is_center': merged_is_center
    }


def phrase_clustering(phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cluster phrases with the same visual properties (font, size, bold, all_cap).
    
    Args:
        phrases (List[Dict[str, Any]]): List of merged phrase objects
        
    Returns:
        List[Dict[str, Any]]: List of cluster objects, each containing phrases with same visual properties
    """
    if not phrases:
        return []
    
    # Create clusters based on visual properties
    clusters = {}
    
    for i, phrase in enumerate(phrases):
        # Create a key for visual properties
        visual_key = (
            phrase['font'],
            phrase['size'],
            phrase['bold'],
            phrase['is_underline'],
            phrase['all_cap']
        )
        
        # Add extraction order to phrase for ranking
        phrase_with_order = phrase.copy()
        phrase_with_order['extraction_order'] = i
        
        if visual_key not in clusters:
            clusters[visual_key] = {
                'visual_properties': {
                    'font': phrase['font'],
                    'size': phrase['size'],
                    'bold': phrase['bold'],
                    'is_underline': phrase['is_underline'],
                    'all_cap': phrase['all_cap']
                },
                'phrases': []
            }
        
        clusters[visual_key]['phrases'].append(phrase_with_order)
    
    # Convert clusters to list and sort by earliest extraction order
    cluster_list = []
    for visual_key, cluster_data in clusters.items():
        # Find the earliest extraction order in this cluster
        earliest_order = min(phrase['extraction_order'] for phrase in cluster_data['phrases'])
        
        # Find the smallest ID in this cluster
        smallest_id = min(phrase['id'] for phrase in cluster_data['phrases'])
        
        # Remove extraction_order from phrases (clean up)
        for phrase in cluster_data['phrases']:
            del phrase['extraction_order']
        
        cluster_list.append({
            'cluster_id': len(cluster_list),
            'earliest_extraction_order': earliest_order,
            'smallest_id': smallest_id,
            'visual_properties': cluster_data['visual_properties'],
            'phrases': cluster_data['phrases']
        })
    
    # Sort clusters by earliest extraction order
    cluster_list.sort(key=lambda x: x['earliest_extraction_order'])
    
    # Remove the sorting key from final output but keep the smallest_id
    for cluster in cluster_list:
        del cluster['earliest_extraction_order']
    
    return cluster_list


def save_clusters_to_json(clusters: List[Dict[str, Any]], output_file: str):
    """Save clustered phrases to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)


def tree_gen(filtered_clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate a hierarchical tree structure from filtered clusters.
    
    Args:
        filtered_clusters (List[Dict[str, Any]]): List of filtered cluster objects
        
    Returns:
        List[Dict[str, Any]]: List of tree nodes with hierarchical structure
    """
    if not filtered_clusters:
        return []
    
    # Create a virtual root node
    root_node = {
        'id': -1,
        'phrase': '',
        'bbox': (0, 0, 0, 0),
        'page': -1,
        'font': '',
        'size': 0,
        'bold': 0,
        'is_underline': 0,
        'all_cap': 0,
        'num_st': 0,
        'is_center': 0,
        'next': [],
        'height': 0
    }
    
    tree_nodes = [root_node]
    
    # Process clusters one by one
    for cluster_idx, cluster in enumerate(filtered_clusters):
        cluster_id = cluster['cluster_id']
        height = cluster_idx + 1  # Height based on cluster order
        
        print(f"Processing cluster {cluster_id} (index {cluster_idx}, height {height})")
        
        # Create nodes for current cluster
        cluster_nodes = []
        for phrase in cluster['phrases']:
            node = {
                'id': phrase['id'],
                'phrase': phrase['phrase'],
                'bbox': phrase['bbox'],
                'page': phrase['page'],
                'font': phrase['font'],
                'size': phrase['size'],
                'bold': phrase['bold'],
                'is_underline': phrase['is_underline'],
                'all_cap': phrase['all_cap'],
                'num_st': phrase['num_st'],
                'is_center': phrase['is_center'],
                'next': [],
                'height': height
            }
            cluster_nodes.append(node)
            tree_nodes.append(node)
        
        print(f"  Created {len(cluster_nodes)} nodes: {[node['id'] for node in cluster_nodes]}")
        
        # Process edge creation for current cluster nodes
        for node_i in cluster_nodes:
            # Check connections to all existing nodes (from previous clusters and current cluster)
            for other_node in tree_nodes[1:]:  # Skip root
                if node_i['id'] < other_node['id']:  # only connect to nodes with smaller id
                    continue
                
                # Find cluster IDs
                other_cluster_id = None
                for other_cluster in filtered_clusters:
                    for phrase in other_cluster['phrases']:
                        if phrase['id'] == other_node['id']:
                            other_cluster_id = other_cluster['cluster_id']
                            break
                    if other_cluster_id is not None:
                        break
                
                # Condition 1: C_i and C_j are different
                if cluster_id != other_cluster_id:
                    # Condition 2: C_i's id is larger than C_j's id
                    if cluster_id > other_cluster_id:
                        # Condition 3: v_i's id is larger than v_j's id
                        if node_i['id'] > other_node['id']:
                            # Condition 4: No other node within the same height as v_i has a smaller id than v_j
                            no_smaller_in_same_height = True
                            for tree_node in tree_nodes[1:]:  # Skip root
                                if (tree_node['height'] == other_node['height'] and 
                                    tree_node['id'] < node_i['id'] and 
                                    tree_node['id'] > other_node['id']):
                                    no_smaller_in_same_height = False
                                    break
                            
                            # Additional condition: Height constraint - only connect to next level
                            height_ok = (other_node['height']+1 == node_i['height'])
                            
                            if no_smaller_in_same_height and height_ok:
                                other_node['next'].append(node_i['id'])
        
        # Connect root to ALL nodes in the first cluster (smallest cluster ID)
        if cluster_idx == 0:  # Only for the first cluster (smallest cluster_id)
            if cluster_nodes:
                # Connect root to ALL nodes in this cluster
                for node in cluster_nodes:
                    root_node['next'].append(node['id'])
                print(f"Connected root to ALL nodes in cluster {cluster_id}: {[node['id'] for node in cluster_nodes]}")
    
    return tree_nodes


def save_tree_to_json(tree_nodes: List[Dict[str, Any]], output_file: str):
    """Save tree structure to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tree_nodes, f, indent=2, ensure_ascii=False)


from typing import Dict, Any, List, Set

def build_id_index(data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Return a mapping from node id -> node dict."""
    return {n["id"]: n for n in data}

def print_tree(nodes_by_id: Dict[int, Dict[str, Any]], node_id: int, depth: int = 0, visited: Set[int] = None):
    """Recursively print the tree using the id->node mapping."""
    if visited is None:
        visited = set()
    if node_id in visited:
        print("  " * depth + f"- [cycle detected at id={node_id}]")
        return
    visited.add(node_id)

    node = nodes_by_id.get(node_id)
    if node is None:
        print("  " * depth + f"- [missing node id={node_id}]")
        return

    phrase = node.get("phrase") or "ROOT"
    print("  " * depth + f"- {phrase} (id={node_id})")

    for child_id in node.get("next", []):
        print_tree(nodes_by_id, child_id, depth + 1, visited)


def cluster_filter(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter clusters to keep only those that contain titles or headings.
    Uses LLM to determine if phrases in each cluster are primarily titles/headings.
    
    Args:
        clusters (List[Dict[str, Any]]): List of cluster objects
        
    Returns:
        List[Dict[str, Any]]: Filtered list of clusters containing titles/headings
    """
    import random
    import sys
    import os
    
    # Add the parent directory to the path to import the model
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.model import model
    
    filtered_clusters = []
    #cid = 0
    for cluster in clusters:
        #print('cluster id: ', cid)
        #cid += 1
        phrases = cluster['phrases']
        
        num_of_sampled_phrases = 10
        # Randomly sample 5 phrases, or all if less than 5
        if len(phrases) >= num_of_sampled_phrases:
            sampled_phrases = random.sample(phrases, num_of_sampled_phrases)
        else:
            sampled_phrases = phrases
        
        # Concatenate phrases into a context string
        context_phrases = [phrase['phrase'] for phrase in sampled_phrases]
        context_string = "\n".join(context_phrases)
        
        # Check average word count - if greater than 20 words, likely not headings
        word_counts = [len(phrase['phrase'].split()) for phrase in sampled_phrases]
        avg_word_count = sum(word_counts) / len(word_counts)
        
        if avg_word_count > 20:
            print(f"Cluster filtered out due to high word count: {avg_word_count:.1f} words average")
            continue
        
        # Check for "Page" + number pattern - if found, likely not headings
        page_pattern_found = False
        for phrase in sampled_phrases:
            phrase_text = phrase['phrase'].lower()
            if 'page' in phrase_text:
                # Check if there's a number after "page"
                import re
                if re.search(r'page\s+\d+', phrase_text):
                    page_pattern_found = True
                    break
        
        if page_pattern_found:
            print(f"Cluster filtered out due to 'Page + number' pattern")
            continue
        
        # Check for repetitive phrases - if most sampled phrases are identical, filter out
        from collections import Counter
        phrase_texts = [phrase['phrase'].strip() for phrase in phrases]
        phrase_counter = Counter(phrase_texts)
        most_common_count = phrase_counter.most_common(1)[0][1] if phrase_counter else 0
        # print(f"Most common phrase count: {most_common_count}")
        # print(f"Sampled phrases: {sampled_phrases}")
        
        # If more than 50% of sampled phrases are identical, filter out
        if most_common_count / len(sampled_phrases) > 0.5:
            most_common_phrase = phrase_counter.most_common(1)[0][0]
            print(f"Cluster filtered out due to repetitive phrases: '{most_common_phrase}' ({most_common_count}/{len(sampled_phrases)} identical)")
            continue
        
        # Create the prompt
        instruction = """Given a list of phrases extracted from a document, determine whether they are primarily titles or section/subsection headings (as opposed to regular body text). Only return "YES" or "NO". Return "YES" if the majority of the phrases function as titles or headings. Return "NO" if the majority are regular text. Do not add explanations. Phrases:"""

        prompt = (instruction, context_string)

        print(context_string)
        
        try:
            # Call the LLM
            response = model('gpt_4o_mini_azure', prompt)
            
            # Clean the response and check if it's "Y"
            response_clean = response.strip().upper()
            print(response_clean) 
            if 'YES' in response_clean: #majority of phrases are titles/headings
                filtered_clusters.append(cluster)
                #print(f"Cluster {cluster['cluster_id']}: KEPT (Y) - {len(phrases)} phrases")
            # else:
            #     print(f"Cluster {cluster['cluster_id']}: FILTERED (N) - {len(phrases)} phrases")
                
        except Exception as e:
            print(f"Error processing cluster {cluster['cluster_id']}: {e}")
            # In case of error, keep the cluster to be safe
            filtered_clusters.append(cluster)
    
    #print(f"\nFiltering complete: {len(clusters)} clusters -> {len(filtered_clusters)} clusters kept")
    return filtered_clusters


# Example usage and testing
if __name__ == "__main__":
    out_folder = 'out/paper'
    os.makedirs(out_folder, exist_ok=True)
    
    tree_structure_file = f"{out_folder}/tree_structure.json"
    
    # Check if tree structure already exists
    if os.path.exists(tree_structure_file):
        print("Found existing tree structure, loading from file...")
        with open(tree_structure_file, 'r', encoding='utf-8') as f:
            tree_nodes = json.load(f)
        print(f"Loaded {len(tree_nodes)} tree nodes from existing file.")
        
        # Also load other data if available for statistics
        phrases = []
        merged_phrases = []
        clusters = []
        filtered_clusters = []
        
        try:
            with open(f"{out_folder}/extracted_phrases.json", 'r', encoding='utf-8') as f:
                phrases = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open(f"{out_folder}/merged_phrases.json", 'r', encoding='utf-8') as f:
                merged_phrases = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open(f"{out_folder}/clustered_phrases.json", 'r', encoding='utf-8') as f:
                clusters = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open(f"{out_folder}/filtered_clusters.json", 'r', encoding='utf-8') as f:
                filtered_clusters = json.load(f)
        except FileNotFoundError:
            pass
    else:
        print("No existing tree structure found, running full pipeline...")
        # Example usage
        pdf_file = "raw_data/paper/A Lived Informatics Model of Personal Informatics.pdf"
        phrases = phrase_visual_pattern_extraction(pdf_file)

        # Save original phrases to JSON file
        save_phrases_to_json(phrases, f"{out_folder}/extracted_phrases.json")
        
        # Merge phrases with same visual properties
        merged_phrases = phrase_merge(phrases)
        
        # Save merged phrases to JSON file
        save_phrases_to_json(merged_phrases, f"{out_folder}/merged_phrases.json")
        
        # Cluster phrases by visual properties
        clusters = phrase_clustering(merged_phrases)
        
        # Save clusters to JSON file
        save_clusters_to_json(clusters, f"{out_folder}/clustered_phrases.json")
        
        # Filter clusters to keep only titles/headings
        filtered_clusters = cluster_filter(clusters)
        
        # Save filtered clusters to JSON file
        save_clusters_to_json(filtered_clusters, f"{out_folder}/filtered_clusters.json")
        
        # Generate tree structure from filtered clusters
        tree_nodes = tree_gen(filtered_clusters)
        
        # Save tree structure to JSON file
        save_tree_to_json(tree_nodes, f"{out_folder}/tree_structure.json")
        print("Full pipeline completed and results saved.")
    
    # Print statistics
    print(f"Original phrases: {len(phrases)}")
    print(f"Merged phrases: {len(merged_phrases)}")
    print(f"Clusters: {len(clusters)}")
    print(f"Filtered clusters: {len(filtered_clusters)}")
    print(f"Tree nodes: {len(tree_nodes)}")
    
    # Print cluster information if available
    # if clusters:
    #     print("\nCluster information:")
    #     for i, cluster in enumerate(clusters[:3]):  # Show first 3 clusters
    #         print(f"Cluster {i}: {cluster['visual_properties']} - {len(cluster['phrases'])} phrases")
    
    # Print filtered cluster information if available
    # if filtered_clusters:
    #     print("\nFiltered cluster information:")
    #     for i, cluster in enumerate(filtered_clusters[:3]):  # Show first 3 filtered clusters
    #         print(f"Filtered Cluster {i}: {cluster['visual_properties']} - {len(cluster['phrases'])} phrases")
    
    # Print tree structure information
    # if tree_nodes:
    #     print("\nTree structure information:")
    #     print(f"Root node: {tree_nodes[0]}")
    #     print(f"Number of levels: {max(node['height'] for node in tree_nodes) if tree_nodes else 0}")
    
    # Print tree visualization
    print("\nTree Visualization:")
    #output_file = 'out/paper/tree_visualization.txt'
    node_ids = build_id_index(tree_nodes)
    print_tree(node_ids, -1)
    

