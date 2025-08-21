import json
from typing import Dict, List, Any
import os
import sys

# Add the core directory to the path to import sht_gen functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sht_gen import (
    phrase_visual_pattern_extraction, phrase_merge, phrase_clustering, 
    save_clusters_to_json, tree_gen, save_phrases_to_json, save_tree_to_json, 
    build_id_index, print_tree, cluster_filter, save_filtered_clusters_to_json
)


def pattern_extraction(tree: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract visual patterns from a tree structure, organized by height.
    Only stores unique patterns for each height.
    
    Args:
        tree: List of tree nodes loaded from tree_structure.json
        
    Returns:
        Dict where key is height and value is a list of unique visual patterns
        Each visual pattern dict contains: 'font', 'size', 'bold', 'is_underline', 'all_cap', 'num_st', 'is_center'
    """
    patterns_by_height = {}
    
    for node in tree:
        # Skip the virtual root node (height 0)
        if node['id'] == -1:
            continue
            
        height = node['height']
        
        # Extract visual pattern from the node
        visual_pattern = {
            'font': node.get('font', 'Unknown'),
            'size': node.get('size', 12),
            'bold': node.get('bold', 0),
            'is_underline': node.get('is_underline', 0),
            'all_cap': node.get('all_cap', 0),
            'num_st': node.get('num_st', 0),
            'is_center': node.get('is_center', 0)
        }
        
        # Initialize height group if not exists
        if height not in patterns_by_height:
            patterns_by_height[height] = []
        
        # Check if this pattern already exists at this height
        # Note: is_center is excluded from duplicate detection
        pattern_exists = False
        for existing_pattern in patterns_by_height[height]:
            if (existing_pattern['font'] == visual_pattern['font'] and
                existing_pattern['size'] == visual_pattern['size'] and
                existing_pattern['bold'] == visual_pattern['bold'] and
                existing_pattern['is_underline'] == visual_pattern['is_underline'] and
                existing_pattern['all_cap'] == visual_pattern['all_cap'] and
                existing_pattern['num_st'] == visual_pattern['num_st']):
                pattern_exists = True
                break
        
        # Only add if pattern doesn't already exist
        if not pattern_exists:
            patterns_by_height[height].append(visual_pattern)
    
    return patterns_by_height


def load_tree_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load tree structure from a JSON file.
    
    Args:
        file_path: Path to the tree_structure.json file
        
    Returns:
        List of tree nodes
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tree from file: {e}")
        return []


def save_patterns_to_json(patterns: Dict[int, List[Dict[str, Any]]], output_file: str):
    """
    Save extracted patterns to a JSON file.
    
    Args:
        patterns: Dictionary of patterns by height
        output_file: Path to save the patterns
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
        print(f"Patterns saved to {output_file}")
    except Exception as e:
        print(f"Error saving patterns: {e}")


def analyze_patterns(patterns: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze the extracted patterns to provide insights.
    
    Args:
        patterns: Dictionary of patterns by height
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_heights': len(patterns),
        'height_distribution': {},
        'font_usage': {},
        'size_distribution': {},
        'bold_usage': {},
        'center_usage': {}
    }
    
    for height, pattern_list in patterns.items():
        analysis['height_distribution'][height] = len(pattern_list)
        
        for pattern in pattern_list:
            # Font usage
            font = pattern['font']
            analysis['font_usage'][font] = analysis['font_usage'].get(font, 0) + 1
            
            # Size distribution
            size = pattern['size']
            analysis['size_distribution'][size] = analysis['size_distribution'].get(size, 0) + 1
            
            # Bold usage
            if pattern['bold'] == 1:
                analysis['bold_usage'][height] = analysis['bold_usage'].get(height, 0) + 1
            
            # Center usage
            if pattern['is_center'] == 1:
                analysis['center_usage'][height] = analysis['center_usage'].get(height, 0) + 1
    
    return analysis


def template_match(tree: List[Dict[str, Any]], doc_path: str) -> tuple[bool, Dict[str, Any]]:
    """
    Match patterns from a tree template with patterns from a document.
    
    Args:
        tree: List of tree nodes (template)
        doc_path: Path to the document PDF file
        
    Returns:
        Tuple of (is_valid, template_match_results)
        - is_valid: True if document matches template pattern, False otherwise
        - template_match_results: Dictionary containing all matching results
    """
    print("=== Template Matching Process ===")
    
    # Step 1: Extract patterns from tree template
    print("1. Extracting patterns from tree template...")
    template_patterns = pattern_extraction(tree)
    print(f"   Template has {len(template_patterns)} heights")
    
    # Step 2: Process document to get clustered phrases
    print("2. Processing document...")
    try:
        # Extract phrases from document
        doc_phrases = phrase_visual_pattern_extraction(doc_path)
        print(f"   Extracted {len(doc_phrases)} phrases from document")
        save_phrases_to_json(doc_phrases, "out/paper/new_extracted_phrases.json")
        
        # Merge phrases
        merged_phrases = phrase_merge(doc_phrases)
        print(f"   Merged into {len(merged_phrases)} phrases")
        save_phrases_to_json(merged_phrases, "out/paper/new_merged_phrases.json")
        
        # Cluster phrases
        clustered_phrases = phrase_clustering(merged_phrases)
        print(f"   Clustered into {len(clustered_phrases)} clusters")
        save_clusters_to_json(clustered_phrases, "out/paper/new_clustered_phrases.json")
        
    except Exception as e:
        print(f"   Error processing document: {e}")
        return False
    
    # Step 3: Create doc_pattern dict by matching template patterns with clustered phrases
    print("3. Creating doc_pattern dict...")
    doc_pattern = {}
    
    for template_height, template_pattern_list in template_patterns.items():
        for template_pattern in template_pattern_list:
            # Check if this template pattern exists in any of the clustered phrases
            for cluster in clustered_phrases:
                for phrase in cluster['phrases']:
                    # Compare patterns (excluding is_center)
                    if (template_pattern['font'] == phrase.get('font', 'Unknown') and
                        template_pattern['size'] == phrase.get('size', 12) and
                        template_pattern['bold'] == phrase.get('bold', 0) and
                        template_pattern['is_underline'] == phrase.get('is_underline', 0) and
                        template_pattern['all_cap'] == phrase.get('all_cap', 0)):
                        
                        # Add P and its corresponding height to doc_pattern
                        # Convert pattern to tuple for hashing
                        pattern_key = (
                            template_pattern['font'],
                            template_pattern['size'],
                            template_pattern['bold'],
                            template_pattern['is_underline'],
                            template_pattern['all_cap'],
                            template_pattern['num_st'],
                            template_pattern['is_center']
                        )
                        doc_pattern[pattern_key] = template_height
                        break  # Found a match, move to next template pattern
                else:
                    continue  # Continue to next cluster if no match found in current cluster
                break  # Found a match, move to next template pattern
    print(doc_pattern)
    #print(f"   Found {len(doc_pattern)} matching patterns")
    
    # Step 4: Check if doc_pattern heights form a valid prefix sequence
    print("4. Checking height sequence validity...")
    doc_heights = set(doc_pattern.values())  # Extract all distinct heights
    print(f"   Document heights: {sorted(doc_heights)}")
    
    # Check if heights form a prefix of 1 to n
    if doc_heights:
        max_height = max(doc_heights)
        expected_heights = set(range(1, max_height + 1))
        
        is_valid_prefix = doc_heights == expected_heights
        print(f"   Expected heights (1 to {max_height}): {sorted(expected_heights)}")
        print(f"   Is valid prefix: {is_valid_prefix}")
        
        if is_valid_prefix:
            print(f"   ✓ Document pattern matches template structure")
        else:
            print(f"   ✗ Document pattern does not match template structure")
            missing_heights = expected_heights - doc_heights
            if missing_heights:
                print(f"   Missing heights: {sorted(missing_heights)}")
    else:
        is_valid_prefix = False
        print(f"   ✗ No matching patterns found")
    
    # Create template match results
    template_match_results = {
        'template_patterns': template_patterns,
        'doc_pattern': doc_pattern,
        'clustered_phrases': clustered_phrases,
        'is_valid_prefix': is_valid_prefix,
        'doc_heights': sorted(doc_heights) if doc_heights else []
    }
    
    return is_valid_prefix, template_match_results


def tree_gen_template(clustered_phrases: List[Dict[str, Any]], doc_pattern: Dict, template_patterns: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Generate tree from filtered phrases based on template matching.
    
    Args:
        clustered_phrases: Original clustered phrases from document
        doc_pattern: Dictionary of matched patterns with heights
        template_patterns: Template patterns extracted from tree
        
    Returns:
        List of tree nodes
    """
    print("=== Tree Generation from Template ===")
    
    # Step 5: Create filtered_phrases from clustered_phrases
    print("5. Creating filtered phrases...")
    filtered_phrases = []
    
    # Get the pattern keys from doc_pattern for matching
    doc_pattern_keys = set(doc_pattern.keys())
    
    for cluster in clustered_phrases:
        if cluster['phrases']:
            # Use the first phrase in cluster to represent the cluster's pattern
            first_phrase = cluster['phrases'][0]
            
            # Create pattern key for comparison (same format as doc_pattern keys)
            cluster_pattern_key = (
                first_phrase.get('font', 'Unknown'),
                first_phrase.get('size', 12),
                first_phrase.get('bold', 0),
                first_phrase.get('is_underline', 0),
                first_phrase.get('all_cap', 0),
                first_phrase.get('num_st', 0),
                first_phrase.get('is_center', 0)
            )
            
            # Check if this cluster's pattern matches any pattern in doc_pattern
            if cluster_pattern_key in doc_pattern_keys:
                filtered_phrases.append(cluster)
    
    print(f"   Original clusters: {len(clustered_phrases)}")
    print(f"   Filtered clusters: {len(filtered_phrases)}")
    save_clusters_to_json(filtered_phrases, "out/paper/new_filtered_phrases.json")
    
    # Step 6: Generate tree from filtered phrases
    print("6. Generating tree from filtered phrases...")
    try:
        tree_nodes = tree_gen(filtered_phrases)
        print(f"   Generated {len(tree_nodes)} tree nodes")
        
        # Save tree to local file
        tree_output_file = "out/paper/new_tree_structure.json"
        save_tree_to_json(tree_nodes, tree_output_file)
        print(f"   Tree saved to: {tree_output_file}")

        print("\nTree Visualization:")
        node_ids = build_id_index(tree_nodes)
        print_tree(node_ids, -1)
        
        return tree_nodes
        
    except Exception as e:
        print(f"   Error generating tree: {e}")
        return []


def sht_clustering(doc_paths: List[str], output_dir: str = "out/clustering") -> Dict[str, Any]:
    """
    Cluster documents by maintaining a pool of distinct templates.
    
    Args:
        doc_paths: List of document PDF file paths
        output_dir: Directory to save clustering results
        
    Returns:
        Dictionary containing clustering results and statistics
    """
    print("=== SHT Clustering Process ===")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize template pool
    template_pool = []
    clustering_results = {
        'total_docs': len(doc_paths),
        'template_pool_size': 0,
        'docs_processed': 0,
        'docs_matched': 0,
        'docs_new_template': 0,
        'doc_assignments': {},  # doc_path -> template_index
        'template_sources': []  # List of template source documents
    }
    
    for i, doc_path in enumerate(doc_paths):
        print(f"\n--- Processing Document {i+1}/{len(doc_paths)}: {os.path.basename(doc_path)} ---")
        
        try:
            # Check if document matches any existing template
            matched_template_index = -1
            matched_template = None
            
            for template_idx, template in enumerate(template_pool):
                print(f"  Checking against template {template_idx + 1}...")
                is_valid, _ = template_match(template, doc_path)
                
                if is_valid:
                    print(f"  ✓ Document matches template {template_idx + 1}")
                    matched_template_index = template_idx
                    matched_template = template
                    clustering_results['docs_matched'] += 1
                    break
                else:
                    print(f"  ✗ Document does not match template {template_idx + 1}")
            
            if matched_template_index >= 0:
                # Document matches existing template - generate tree using template
                print(f"  Generating tree using existing template {matched_template_index + 1}...")
                
                # Get the template match results
                _, template_results = template_match(matched_template, doc_path)
                
                # Generate tree using template
                tree_nodes = tree_gen_template(
                    template_results['clustered_phrases'],
                    template_results['doc_pattern'],
                    template_results['template_patterns']
                )
                
                # Save tree with template reference
                tree_output_file = os.path.join(output_dir, f"doc_{i+1}_template_{matched_template_index+1}_tree.json")
                save_tree_to_json(tree_nodes, tree_output_file)
                
                clustering_results['doc_assignments'][doc_path] = {
                    'template_index': matched_template_index,
                    'template_source': clustering_results['template_sources'][matched_template_index],
                    'tree_file': tree_output_file,
                    'method': 'template_match'
                }
                
            else:
                # Document doesn't match any template - create new template
                print(f"  Creating new template for document...")
                
                # Process document using full sht_gen pipeline
                doc_phrases = phrase_visual_pattern_extraction(doc_path)
                merged_phrases = phrase_merge(doc_phrases)
                clustered_phrases = phrase_clustering(merged_phrases)
                filtered_clusters = cluster_filter(clustered_phrases)
                new_template = tree_gen(filtered_clusters)
                
                # Add new template to pool
                template_pool.append(new_template)
                clustering_results['template_sources'].append(doc_path)
                clustering_results['docs_new_template'] += 1
                
                # Save new template
                template_output_file = os.path.join(output_dir, f"template_{len(template_pool)}_source_{os.path.basename(doc_path)}.json")
                save_tree_to_json(new_template, template_output_file)
                
                # Save document's tree (same as template for new documents)
                tree_output_file = os.path.join(output_dir, f"doc_{i+1}_new_template_tree.json")
                save_tree_to_json(new_template, tree_output_file)
                
                clustering_results['doc_assignments'][doc_path] = {
                    'template_index': len(template_pool) - 1,
                    'template_source': doc_path,
                    'tree_file': tree_output_file,
                    'method': 'new_template'
                }
                
                print(f"  ✓ Created new template {len(template_pool)}")
            
            clustering_results['docs_processed'] += 1
            
        except Exception as e:
            print(f"  ✗ Error processing document: {e}")
            clustering_results['doc_assignments'][doc_path] = {
                'template_index': -1,
                'template_source': None,
                'tree_file': None,
                'method': 'error',
                'error': str(e)
            }
    
    # Update final statistics
    clustering_results['template_pool_size'] = len(template_pool)
    
    # Save clustering results
    results_file = os.path.join(output_dir, "clustering_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(clustering_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== Clustering Summary ===")
    print(f"Total documents processed: {clustering_results['docs_processed']}")
    print(f"Documents matched to existing templates: {clustering_results['docs_matched']}")
    print(f"Documents requiring new templates: {clustering_results['docs_new_template']}")
    print(f"Final template pool size: {clustering_results['template_pool_size']}")
    print(f"Results saved to: {results_file}")
    
    return clustering_results


def print_pattern_summary(patterns: Dict[int, List[Dict[str, Any]]]):
    """
    Print a summary of the extracted patterns.
    
    Args:
        patterns: Dictionary of patterns by height
    """
    print("=== Pattern Extraction Summary ===")
    print(f"Total heights: {len(patterns)}")
    
    for height in sorted(patterns.keys()):
        pattern_list = patterns[height]
        print(f"\nHeight {height}: {len(pattern_list)} nodes")
        
        # Show unique patterns at this height
        unique_patterns = {}
        for pattern in pattern_list:
            pattern_key = f"{pattern['font']}_{pattern['size']}_{pattern['bold']}_{pattern['is_underline']}_{pattern['all_cap']}_{pattern['num_st']}_{pattern['is_center']}"
            unique_patterns[pattern_key] = unique_patterns.get(pattern_key, 0) + 1
        
        print(f"  Unique patterns: {len(unique_patterns)}")
        
        # Show most common patterns
        sorted_patterns = sorted(unique_patterns.items(), key=lambda x: x[1], reverse=True)
        for i, (pattern_key, count) in enumerate(sorted_patterns[:3]):  # Show top 3
            if i == 0:
                print(f"  Most common: {pattern_key} (count: {count})")


# Example usage
if __name__ == "__main__":
    # Example: Load tree from a file and extract patterns
    tree_file = "out/paper/tree_structure.json"
    
    if os.path.exists(tree_file):
        print(f"Loading tree from {tree_file}")
        tree = load_tree_from_file(tree_file)
        
        if tree:
            print(f"Loaded {len(tree)} tree nodes")
            
            # Extract patterns
            patterns = pattern_extraction(tree)
            
            # Print summary
            print_pattern_summary(patterns)
            
            # Save patterns to file
            output_file = "out/paper/visual_patterns.json"
            save_patterns_to_json(patterns, output_file)
            
            # Analyze patterns
            analysis = analyze_patterns(patterns)
            print(f"\nAnalysis:")
            print(f"  Total heights: {analysis['total_heights']}")
            print(f"  Height distribution: {analysis['height_distribution']}")
            print(f"  Most common fonts: {dict(sorted(analysis['font_usage'].items(), key=lambda x: x[1], reverse=True)[:3])}")
            print(f"  Most common sizes: {dict(sorted(analysis['size_distribution'].items(), key=lambda x: x[1], reverse=True)[:3])}")
            
            # Example template matching
            print("\n" + "="*50)
            print("EXAMPLE TEMPLATE MATCHING")
            print("="*50)
            
            # Use the same document as template for demonstration
            doc_path = "raw_data/paper/A Lived Informatics Model of Personal Informatics.pdf"
            
            if os.path.exists(doc_path):
                print(f"Running template matching with document: {doc_path}")
                match_results = template_match(tree, doc_path)
                
                
