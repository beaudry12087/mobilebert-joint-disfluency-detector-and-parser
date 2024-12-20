import torch
import sys
import os
from collections import defaultdict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
import parse_nk
import trees


 # Test sentences
from test_sentences import test_sentences
import torch
import sys
import os
from collections import defaultdict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
import parse_nk
import trees

class DisfluencyValidator:
    def __init__(self):
        self.GUARANTEED_PATTERNS = {
            'numbers': ['nine one one', '911']
        }
        self.metrics = defaultdict(int)
    
    def analyze_output(self, text, disfluencies):
        """Validate model outputs against known patterns"""
        results = {
            'potential_errors': [],
            'model_predictions': []
        }
        
        if not disfluencies:
            return results
            
        words = text.lower().split()
        
        # Check for number sequences
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if any(pattern in trigram for pattern in self.GUARANTEED_PATTERNS['numbers']):
                if i+2 < len(disfluencies) and any(disfluencies[j][1] for j in range(i, i+3)):
                    results['potential_errors'].append({
                        'span': trigram,
                        'type': 'number_sequence',
                        'confidence': 'high'
                    })
                    self.metrics['number_errors'] += 1
        
        return results

def torch_load(load_path):
    """Load PyTorch model with CPU/GPU handling"""
    if torch.cuda.is_available():
        return torch.load(load_path)
    return torch.load(load_path, map_location=lambda storage, location: storage)

def analyze_disfluencies(tree, ignore_agreement_fillers=True):
    """Analyze tree structure to identify disfluencies"""
    disfluencies = []
    
    # Core detection sets
    AGREEMENT_SOUNDS = {'mhmm', 'mm-mm', 'uh-huh', 'uh-uh', 'nuh-uh'}
    FILLERS = {'uh', 'um', 'er', 'ah', 'mm'}
    
    def traverse(node, in_edited=False, in_intj=False):
        if isinstance(node, trees.LeafTreebankNode):
            word = node.word
            word_clean = word.lower().rstrip(',.!?')
            
            # Check for agreement sounds
            near_agreement = False
            if ignore_agreement_fillers and len(disfluencies) > 0:
                prev_word, _ = disfluencies[-1]
                if prev_word.lower().rstrip(',.!?') in AGREEMENT_SOUNDS:
                    near_agreement = True
            
            is_filler = word_clean in FILLERS and not near_agreement
            is_disf = in_edited or in_intj or is_filler
            
            disfluencies.append((word, is_disf))
            return
            
        is_edited = in_edited or (hasattr(node, 'label') and 'EDITED' in str(node.label))
        is_intj = in_intj or (hasattr(node, 'label') and 'INTJ' in str(node.label))
        
        if hasattr(node, 'children'):
            for child in node.children:
                traverse(child, is_edited, is_intj)
    
    traverse(tree)
    return disfluencies

def format_disfluencies(disfluencies, merge_consecutive=True):
    """Format disfluencies for display, optionally merging consecutive markers"""
    if not merge_consecutive:
        return ' '.join('[' + word + ']' if is_disf else word 
                       for word, is_disf in disfluencies)
    
    result = []
    current_group = []
    last_was_disf = False
    
    for i, (word, is_disf) in enumerate(disfluencies):
        # Always merge if:
        # 1. Current word is disfluent and we have an active group
        # 2. Current word is disfluent and previous word was disfluent
        # 3. Current word is a filler (um, uh) and near other fillers
        # 4. Current word is part of a repair sequence
        if is_disf:
            if (current_group or last_was_disf or 
                word.lower().strip(',.!?') in {'um', 'uh'} or
                (i > 0 and i < len(disfluencies)-1)):  # Part of sequence
                current_group.append(word)
            else:
                if current_group:
                    result.append(f"[{' '.join(current_group)}]")
                current_group = [word]
            last_was_disf = True
        else:
            if current_group:
                result.append(f"[{' '.join(current_group)}]")
                current_group = []
            result.append(word)
            last_was_disf = False
    
    if current_group:
        result.append(f"[{' '.join(current_group)}]")
    
    return ' '.join(result)


def process_sentence(parser, text, validator=None, ignore_agreement_fillers=True, merge_consecutive=False):
    """Process a single sentence and return disfluencies"""
    if len(text.split()) <= 2:
        return None
        
    words = text.split()
    dummy_tag = 'UNK' if 'UNK' in parser.tag_vocab.indices else parser.tag_vocab.value(0)
    sentence = [(dummy_tag, word) for word in words]
    
    try:
        predicted_trees, _ = parser.parse_batch([sentence], compute_reparandum=True)
        
        if predicted_trees:
            tree = predicted_trees[0].convert()
            disfluencies = analyze_disfluencies(tree, ignore_agreement_fillers)
            
            # Validate if validator is provided
            if validator:
                validation_results = validator.analyze_output(text, disfluencies)
                if validation_results['potential_errors']:
                    for error in validation_results['potential_errors']:
                        print(f"\nPotential error: {error['span']} ({error['type']})")
            
            # Return formatted text if disfluencies found
            if any(is_disf for _, is_disf in disfluencies):
                #print("\n=== Processing Disfluencies ===")
                #print(f"Input text: {text[:100]}...")  # Debug log
                #print(f"Number of disfluencies: {len([d for d, is_disf in disfluencies if is_disf])}")
                
                # Use format_disfluencies instead of simple list comprehension
                return format_disfluencies(disfluencies, merge_consecutive=merge_consecutive)
                
    except Exception as e:
        print(f"Error processing sentence: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def test_model(test_sentences):
    """Main testing function"""
    print("Loading model checkpoint...")
    info = torch_load("mobilebert_model_dev=0.8976_rep=0.8185_int=0.9767.pt")
    
    parser = parse_nk.MobileBERTChartParser.from_spec(info['spec'], info['state_dict'])
    parser.eval()
    
    validator = DisfluencyValidator()
    
    for text in test_sentences:
        result = process_sentence(parser, text, validator)
        if result:
            print(result)
    
    # Print statistics
    print(f"\nNumber sequence errors: {validator.metrics['number_errors']}")

if __name__ == "__main__":
    # Import test sentences from separate file or define them here
    test_model(test_sentences)