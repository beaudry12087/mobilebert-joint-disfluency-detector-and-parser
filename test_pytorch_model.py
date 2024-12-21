import torch
import sys
import os
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def analyze_disfluencies(tree, ignore_agreement_fillers=True, use_discourse_detection=True):
    """Analyze tree structure to identify disfluencies with improved protection zones"""
    disfluencies = []
    protected_indices = set()  # Global set of protected indices
    protected_ranges = []  # Track ranges of protected indices
    
    # Core detection sets
    AGREEMENT_SOUNDS = {'mhmm', 'mm-mm', 'uh-huh', 'uh-uh', 'nuh-uh'}
    FILLERS = {'uh', 'um', 'er', 'ah', 'mm'}
    
    # Discourse marker patterns
    DISCOURSE_MARKERS = {
        'and': {
            'valid_prev': {'.', '!', '?'},  # Current
            'valid_narrative_start': True,   # New: Allow some sentence-initial cases
            'valid_next': {'then', 'also', 'finally'}  # New: Common valid sequences
        },
        'like': {
            'valid_prev': {'feel', 'feels', 'felt', 'seem', 'seems', 'seemed', 'look', 'looks', 'looked'},
            'valid_next': {'a', 'an', 'the', 'that', 'this', 'when', 'if'},
            'invalid_pos': {'SENT_START'}
        }
    }
    
    def is_valid_discourse_marker(word, prev_word, next_word, marker_type, current_index):
        """Check if a potential discourse marker is being used validly with logging"""
        if not use_discourse_detection:
            return False
            
        if marker_type not in DISCOURSE_MARKERS:
            return False
            
        patterns = DISCOURSE_MARKERS[marker_type]
        context = f"(prev: '{prev_word}', current: '{word}', next: '{next_word}')"

        if not prev_word and patterns.get('valid_narrative_start', False):
            return True
        
        if marker_type == 'like':
            # Clean words for comparison (more permissive with commas)
            prev_clean = prev_word.lower().rstrip(',.!?')
            next_clean = next_word.lower().rstrip(',.!?')
            
            # 1. Common valid phrases (expanded and prioritized)
            VALID_PHRASES = {
                # High confidence patterns
                'like_i_said': {'prev': '', 'next': {'i', 'you', 'we'}},  # "like I said"
                'modal_like': {'prev': {'would', 'could', 'might', 'will'}, 'next': {'to', 'that', 'if', 'when'}},  # "would like to"
                'verb_like': {'prev': {'feel', 'feels', 'felt', 'seem', 'seems', 'seemed'}, 'next': None},  # "feels like"
                
                # Comparison patterns
                'just_like': {'prev': {'just'}, 'next': None},
                'something_like': {'prev': {'something', 'anything'}, 'next': None},
                'kind_of_like': {'prev': {'kind', 'sort'}, 'next': None},
                
                # Article patterns
                'like_a': {'prev': None, 'next': {'a', 'an', 'the'}},
                'like_this': {'prev': None, 'next': {'this', 'that', 'these', 'those'}}
            }

            # 2. Check for common phrases first
            for phrase_type, patterns in VALID_PHRASES.items():
                prev_match = patterns['prev'] is None or prev_clean in patterns['prev']
                next_match = patterns['next'] is None or next_clean in patterns['next']
                if prev_match and next_match:
                    logger.info(f"Valid 'like' usage found at index {current_index} - {phrase_type} pattern {context}")
                    return True

            # 3. Special handling for comma-separated phrases
            if prev_clean.endswith(','):
                # Look for valid following patterns
                if next_clean in {'i', 'you', 'we', 'they', 'the', 'a', 'an', 'this', 'that', 'when', 'if', 'how'}:
                    logger.info(f"Valid 'like' usage found at index {current_index} - comma separated comparison {context}")
                    return True
                # Look for verb phrases after comma
                if next_clean in {'be', 'do', 'get', 'have', 'make', 'take', 'see', 'know', 'think', 'want'}:
                    logger.info(f"Valid 'like' usage found at index {current_index} - comma separated verb phrase {context}")
                    return True

            # 4. Check for numerical comparisons and proper nouns
            if next_word and (
                next_word.replace(',', '').isdigit() or 
                next_word[0].isupper() or
                any(w.isdigit() for w in next_word.split())
            ):
                logger.info(f"Valid 'like' usage found at index {current_index} - numerical/proper noun comparison {context}")
                return True

            # 5. Better logging for invalid cases
            if not prev_clean and not next_clean:
                logger.info(f"Invalid 'like' usage - isolated usage at index {current_index} {context}")
            elif prev_clean in {'um', 'uh', 'er'} or next_clean in {'um', 'uh', 'er'}:
                logger.info(f"Invalid 'like' usage - adjacent to filler words at index {current_index} {context}")
            else:
                logger.info(f"Invalid 'like' usage - no valid pattern match at index {current_index} {context}")
            return False
            
        if marker_type == 'and':
            if not prev_word:
                logger.info(f"Invalid 'and' usage detected at index {current_index} - no previous word {context}")
                return False
            
            if len(prev_word) > 0 and prev_word[-1] in patterns['valid_prev']:
                logger.info(f"Valid 'and' usage found at index {current_index} - starts new sentence {context}")
                return True
            
            # Check invalid_next only if the pattern exists
            if 'invalid_next' in patterns and next_word.lower() in patterns['invalid_next']:
                logger.info(f"Invalid 'and' usage detected at index {current_index} - followed by conjunction {context}")
                return False
            
            logger.info(f"Valid 'and' usage found at index {current_index} - normal conjunction {context}")
            return True
            
        return False
    
    
    
    def is_potential_disfluency(word):
        """Check if word might be marked as disfluent"""
        word_clean = word.lower().rstrip(',.!?')
        return (word_clean in FILLERS or 
                word_clean in DISCOURSE_MARKERS or 
                word_clean in {'the', 'a', 'an'})  # Common repeats
    
    def get_next_word(node, parent):
        """Get the next word in the tree by looking ahead.
        
        Args:
            node: Current LeafTreebankNode
            parent: Parent node containing the current node
            
        Returns:
            str: Next word if found, empty string otherwise
        """
        if not parent or not parent.children:
            return ''
        
        try:
            current_idx = parent.children.index(node)
            if current_idx < len(parent.children) - 1:
                next_sibling = parent.children[current_idx + 1]
                # Traverse until we find the next leaf node
                while hasattr(next_sibling, 'children') and next_sibling.children:
                    next_sibling = next_sibling.children[0]
                if isinstance(next_sibling, trees.LeafTreebankNode):
                    return next_sibling.word
        except ValueError:  # node not found in parent's children
            pass
        
        return ''
    
    def get_words_ahead(node, parent, count=3):
        """Get the next few words from the tree"""
        words = []
        if not parent or not parent.children:
            return words
            
        try:
            current_idx = parent.children.index(node)
            next_idx = current_idx + 1
            while len(words) < count and next_idx < len(parent.children):
                next_node = parent.children[next_idx]
                while hasattr(next_node, 'children') and next_node.children:
                    next_node = next_node.children[0]
                if isinstance(next_node, trees.LeafTreebankNode):
                    words.append(next_node.word)
                next_idx += 1
        except ValueError:
            pass
        return words

    def is_emphasis_repetition(words):
        """Enhanced check for intentional emphasis patterns"""
        if len(words) < 2:
            return False
        
        # Clean words for comparison
        clean_words = [w.lower().rstrip(',.!?') for w in words]
        
        # Expanded patterns that suggest emphasis
        EMPHASIS_PATTERNS = {
            # Intensifiers
            'very', 'really', 'so', 'just', 'many', 'much', 'big', 'huge',
            # Agreement/Disagreement
            'no', 'yes', 'yeah', 'right', 'true', 'sure',
            # Personal pronouns (often used for emphasis)
            'i', 'you', 'we', 'they',
            # Demonstratives
            'this', 'that', 'these', 'those'
        }
        
        # Check for common emphasis contexts
        EMPHASIS_CONTEXTS = {
            'so', 'just', 'really', 'very',
            'please', 'do', 'must', 'need'
        }
        
        # If it's a pronoun or demonstrative with exactly two repetitions
        if len(clean_words) == 2 and clean_words[0] == clean_words[1]:
            if clean_words[0] in {'i', 'you', 'we', 'they', 'this', 'that', 'these', 'those'}:
                logger.info(f"Detected pronoun/demonstrative emphasis: '{words[0]} {words[1]}'")
                return True
        
        # Check surrounding context for emphasis markers
        context = ' '.join(clean_words)
        if any(marker in context for marker in EMPHASIS_CONTEXTS):
            logger.info(f"Detected emphasis context: '{context}'")
            return True
        
        # Check if the repeated word is commonly used for emphasis
        if clean_words[0] in EMPHASIS_PATTERNS:
            logger.info(f"Detected emphasis word: '{words[0]}'")
            return True
            
        return False

    def protect_surrounding_words(index, current_words, next_words, direction='both'):
        """Enhanced protection zone with tree-based lookahead"""
        logger.info(f"\nProtecting words around index {index}")
        
        if direction in ['both', 'backward'] and index > 0:
            prev_word = current_words[index-1][0] if isinstance(current_words[index-1], tuple) else current_words[index-1]
            protected_indices.add(index - 1)
            logger.info(f"Protected backward: index {index-1} ('{prev_word}')")
        
        if direction in ['both', 'forward'] and next_words:
            # Check if next words form a repetition
            if len(next_words) >= 2 and next_words[0].lower().rstrip(',.!?') == next_words[1].lower().rstrip(',.!?'):
                # Check if it's an emphasis pattern
                if is_emphasis_repetition(next_words):
                    # Protect the repetition sequence
                    protected_indices.add(index + 1)
                    protected_indices.add(index + 2)
                    logger.info(f"Protected emphasis repetition: '{next_words[0]} {next_words[1]}'")
                    return
                
                logger.info(f"Found disfluent repetition sequence starting with '{next_words[0]}'")
                return
            
            # Protect next word if not part of repetition
            protected_indices.add(index + 1)
            logger.info(f"Protected forward: index {index + 1} ('{next_words[0]}')")

    def traverse(node, in_edited=False, in_intj=False, parent=None):
        if isinstance(node, trees.LeafTreebankNode):
            word = node.word
            word_clean = word.lower().rstrip(',.!?')
            current_index = len(disfluencies)
            
            # Get context
            prev_word = disfluencies[-1][0] if disfluencies else ''
            next_word = get_next_word(node, parent)
            
            # Check for valid discourse marker
            is_discourse_marker = False
            if use_discourse_detection and word_clean in DISCOURSE_MARKERS:
                is_discourse_marker = is_valid_discourse_marker(
                    word_clean, prev_word, next_word, word_clean, current_index
                )
                if is_discourse_marker:
                    next_words = get_words_ahead(node, parent)
                    protect_surrounding_words(current_index, disfluencies, next_words) # Now updates global set
            
            # Determine if word is disfluent
            is_disf = (in_edited or 
                      in_intj or 
                      word_clean in FILLERS or 
                      (not is_discourse_marker and word_clean in DISCOURSE_MARKERS))
            
            # Check protection status
            if current_index in protected_indices:
                if is_disf:
                    logger.info(f"Prevented disfluency marking at index {current_index}: '{word}' (protected)")
                is_disf = False
            elif is_disf:
                # Check if in any protected range
                for start, end in protected_ranges:
                    if start <= current_index < end:
                        logger.info(f"Prevented consecutive disfluency at index {current_index}: '{word}'")
                        is_disf = False
                        break
                
                if is_disf:
                    logger.info(f"Marked as disfluent at index {current_index}: '{word}'")
            
            disfluencies.append((word, is_disf))
            return
            
        is_edited = in_edited or (hasattr(node, 'label') and 'EDITED' in str(node.label))
        is_intj = in_intj or (hasattr(node, 'label') and 'INTJ' in str(node.label))
        
        if hasattr(node, 'children'):
            for child in node.children:
                traverse(child, is_edited, is_intj, node)
    
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


def process_sentence(parser, text, validator=None, ignore_agreement_fillers=True, 
                    merge_consecutive=False, use_discourse_detection=True):
    """Process a single sentence with logging"""
    logger.info(f"\nProcessing sentence: {text[:100]}...")
    #logger.info(f"Discourse detection: {'enabled' if use_discourse_detection else 'disabled'}")
    
    if len(text.split()) <= 2:
        #logger.info("Sentence too short, skipping")
        return None
        
    words = text.split()
    dummy_tag = 'UNK' if 'UNK' in parser.tag_vocab.indices else parser.tag_vocab.value(0)
    sentence = [(dummy_tag, word) for word in words]
    
    try:
        predicted_trees, _ = parser.parse_batch([sentence], compute_reparandum=True)
        
        if predicted_trees:
            tree = predicted_trees[0].convert()
            disfluencies = analyze_disfluencies(
                tree, 
                ignore_agreement_fillers=ignore_agreement_fillers,
                use_discourse_detection=use_discourse_detection
            )
            
            if any(is_disf for _, is_disf in disfluencies):
                result = format_disfluencies(disfluencies, merge_consecutive=merge_consecutive)
                logger.info(f"Final output: {result[:100]}...")
                return result
            #else:
                #logger.info("No disfluencies found")
                
    except Exception as e:
        logger.error(f"Error processing sentence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return None

def test_model(test_sentences):
    """Main testing function"""
    print("Loading model checkpoint...")
    info = torch_load("mobilebert_model_dev=0.8976_rep=0.8185_int=0.9767.pt")
    
    parser = parse_nk.MobileBERTChartParser.from_spec(info['spec'], info['state_dict'])
    parser.eval()
    
    validator = DisfluencyValidator()
    stats = {
        'total_sentences': 0,
        'sentences_with_disfluencies': 0,
        'total_disfluencies': 0
    }
   
    for text in test_sentences:
        stats['total_sentences'] += 1
        result = process_sentence(parser, text, validator)
        if result:
            disfluency_count = result.count('[')  # Count opening brackets
            if disfluency_count > 0:
                stats['sentences_with_disfluencies'] += 1
                stats['total_disfluencies'] += disfluency_count
    
    # Print statistics
    print("\nModel Performance Statistics:")
    print(f"Total sentences processed: {stats['total_sentences']}")
    print(f"Sentences with disfluencies: {stats['sentences_with_disfluencies']}")
    print(f"Total disfluencies detected: {stats['total_disfluencies']}")
    print(f"Disfluency rate: {stats['total_disfluencies']/stats['total_sentences']:.2f} per sentence")
    print(f"Number sequence errors: {validator.metrics['number_errors']}")

if __name__ == "__main__":
    # Import test sentences from separate file or define them here
    test_model(test_sentences)