import tensorflow as tf
import numpy as np
from transformers import MobileBertTokenizer
from collections import defaultdict
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from test_pytorch_model import trees  # Import the tree classes

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

def process_output(output_data, text, tokenizer):
    """Process model output to get disfluencies"""
    # Debug: Print raw scores
    label_scores = output_data['label_scores'][0]
    interregnum_scores = output_data['interregnum_scores'][0]
    
    print("\nDebug - Score ranges:")
    print(f"Label scores: min={label_scores.min():.3f}, max={label_scores.max():.3f}")
    print(f"Interregnum scores: min={interregnum_scores.min():.3f}, max={interregnum_scores.max():.3f}")
    
    # Tokenize input
    tokens = tokenizer(text, add_special_tokens=True)
    input_ids = tokens['input_ids']
    
    # Get word-level predictions
    disfluencies = []
    words = text.split()
    current_word_idx = 0
    current_token_idx = 1  # Skip [CLS] token
    
    # Define constants
    FILLERS = {'um', 'uh', 'er', 'ah', 'mm'}
    LABEL_THRESHOLD = 0.75  # Increased from 0.5
    INTERREGNUM_THRESHOLD = 0.75  # Increased from 0.5
    
    # Common words that should rarely be marked as disfluent
    COMMON_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    
    while current_word_idx < len(words) and current_token_idx < len(input_ids):
        word = words[current_word_idx]
        word_tokens = tokenizer.tokenize(word)
        n_tokens = len(word_tokens)
        
        # Get scores for this word
        word_label_scores = label_scores[current_token_idx:current_token_idx + n_tokens]
        word_interregnum_scores = interregnum_scores[current_token_idx:current_token_idx + n_tokens]
        
        # Calculate average scores
        avg_label_score = np.mean(word_label_scores)
        avg_interregnum_score = np.mean(word_interregnum_scores)
        
        # Determine if word is disfluent
        is_filler = word.lower().strip(',.!?') in FILLERS
        has_high_label_score = avg_label_score > LABEL_THRESHOLD
        has_high_interregnum_score = avg_interregnum_score > INTERREGNUM_THRESHOLD
        
        # Add additional checks
        word_lower = word.lower().strip(',.!?')
        is_common_word = word_lower in COMMON_WORDS
        
        # Only mark as disfluent if:
        # 1. It's a known filler word, OR
        # 2. It has very high scores AND is not a common word
        is_disfluent = is_filler or ((has_high_label_score or has_high_interregnum_score) and not is_common_word)
        
        disfluencies.append((word, is_disfluent))
        current_token_idx += n_tokens
        current_word_idx += 1
    
    return disfluencies

def prepare_input(text, tokenizer):
    """Prepare input text for model inference"""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return np.array(tokens, dtype=np.int32)

def process_sentence(interpreter, text, validator=None, ignore_agreement_fillers=True, merge_consecutive=True):
    """Process a single sentence using TFLite model"""
    if len(text.split()) <= 2:
        return None
        
    try:
        # Get interpreter details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Initialize tokenizer
        tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
        
        # Prepare input - tokenize and reshape
        tokens = tokenizer.encode(text, add_special_tokens=True)
        input_data = np.array(tokens, dtype=np.int32)
        
        # Get expected input shape from model
        input_shape = input_details[0]['shape']
        
        # Reshape input to match model's expected shape
        # If model expects (1, 1), we'll need to process tokens one at a time
        if input_shape[1] == 1:
            results = []
            for token in tokens:
                token_input = np.array([[token]], dtype=np.int32)
                interpreter.set_tensor(input_details[0]['index'], token_input)
                interpreter.invoke()
                
                # Get outputs for this token
                label_scores = interpreter.get_tensor(output_details[0]['index'])
                interregnum_scores = interpreter.get_tensor(output_details[1]['index'])
                results.append({
                    'label_scores': label_scores,
                    'interregnum_scores': interregnum_scores
                })
            
            # Combine results
            output_data = {
                'label_scores': np.concatenate([r['label_scores'] for r in results], axis=1),
                'interregnum_scores': np.concatenate([r['interregnum_scores'] for r in results], axis=1)
            }
        else:
            # Process entire sequence at once
            input_data = input_data.reshape(1, -1)  # Add batch dimension
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get outputs
            output_data = {
                'label_scores': interpreter.get_tensor(output_details[0]['index']),
                'interregnum_scores': interpreter.get_tensor(output_details[1]['index'])
            }
        
        # Process output
        disfluencies = process_output(output_data, text, tokenizer)
        
        # Validate if validator is provided
        if validator:
            validation_results = validator.analyze_output(text, disfluencies)
            if validation_results['potential_errors']:
                for error in validation_results['potential_errors']:
                    print(f"\nPotential error: {error['span']} ({error['type']})")
        
        # Return formatted text if disfluencies found
        if any(is_disf for _, is_disf in disfluencies):
            return format_disfluencies(disfluencies, merge_consecutive=merge_consecutive)
            
    except Exception as e:
        print(f"Error processing sentence: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def test_model(test_sentences):
    """Main testing function"""
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path="modelv3.tflite")
    interpreter.allocate_tensors()
    
    validator = DisfluencyValidator()
    
    print("\nProcessing sentences...\n")
    disfluency_count = 0
    
    for text in test_sentences:
        result = process_sentence(interpreter, text, validator)
        if result:  # Only print if disfluencies were found
            disfluency_count += 1
            print(f"Original: {text}")
            print(f"Marked  : {result}")
            print("-" * 80)  # Separator for readability
    
    # Print summary
    print(f"\nFound disfluencies in {disfluency_count} sentences")
    if validator.metrics['number_errors'] > 0:
        print(f"Number sequence errors: {validator.metrics['number_errors']}")

if __name__ == "__main__":
    # Import test sentences from test_pytorch_model
    from test_sentences import test_sentences
    test_model(test_sentences)