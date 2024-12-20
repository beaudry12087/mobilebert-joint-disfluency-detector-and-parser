import tensorflow as tf
import torch
from test_model import process_sentence as process_tflite
from test_pytorch_model import process_sentence as process_pytorch, torch_load
import parse_nk
from transformers import MobileBertTokenizer

def compare_models(test_sentences):
    """Compare outputs from PyTorch and TFLite models"""
    print("Loading models...")
    
    # Load PyTorch model
    info = torch_load("mobilebert_model_dev=0.8976_rep=0.8185_int=0.9767.pt")
    pytorch_parser = parse_nk.MobileBERTChartParser.from_spec(info['spec'], info['state_dict'])
    pytorch_parser.eval()
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="modelv5.tflite")
    interpreter.allocate_tensors()
    
    print("\nProcessing sentences...\n")
    differences = 0
    total_processed = 0
    
    for i, text in enumerate(test_sentences, 1):
        if len(text.split()) <= 2:
            continue
            
        total_processed += 1
        print(f"\nSentence {i}:")
        print(f"Original: {text}")
        
        # Get predictions from both models
        pytorch_result = process_pytorch(pytorch_parser, text, merge_consecutive=True)
        tflite_result = process_tflite(interpreter, text)
        
        # Print results
        print("PyTorch : " + (pytorch_result if pytorch_result else "No disfluencies detected"))
        print("TFLite  : " + (tflite_result if tflite_result else "No disfluencies detected"))
        
        # Check if results differ
        if pytorch_result != tflite_result:
            differences += 1
            print("⚠️ Models disagree on this sentence")
        print("-" * 80)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total sentences processed: {total_processed}")
    print(f"Number of differences: {differences}")
    print(f"Agreement rate: {((total_processed - differences) / total_processed) * 100:.1f}%")

if __name__ == "__main__":
    # Import test sentences
    from test_sentences import test_sentences
    compare_models(test_sentences)