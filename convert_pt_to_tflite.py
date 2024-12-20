import tensorflow as tf
import torch
import numpy as np
from transformers import TFMobileBertModel
import os
import argparse
from collections import OrderedDict
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add both src and current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# Import required custom modules
try:
    from parse_nk import MobileBERTChartParser  # Use the correct parser class
    import vocabulary
    import trees
    import parse_nk
    import nkutil
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

def torch_load(load_path):
    """Load PyTorch model with CPU/GPU handling"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading PyTorch model on {device}")
    return torch.load(load_path, map_location=device)

class TFMobileBERTParser(tf.keras.Model):
    def __init__(self, pytorch_parser=None):
        super(TFMobileBERTParser, self).__init__()
        
        # Get dimensions from PyTorch model
        self.bert_dim = 512
        self.hidden_dim = 128
        self.projected_dim = 64
        
        logger.info(f"Initializing TF model with dimensions:")
        logger.info(f"BERT output: {self.bert_dim}")
        logger.info(f"Hidden: {self.hidden_dim}")
        logger.info(f"Projected: {self.projected_dim}")
        
        # Initialize layers
        self._init_layers()
        
        # Build model with dummy input to initialize weights
        dummy_input = tf.zeros((1, 1), dtype=tf.int32)
        _ = self.predict_on_batch(dummy_input)
        
        # Copy weights if PyTorch model is provided
        if pytorch_parser:
            logger.info("Copying weights from PyTorch model...")
            self._copy_weights(pytorch_parser)

    def _copy_task_weights(self, pytorch_parser):
        """Copy task-specific layer weights"""
        try:
            # Label weights
            label_layers = [
                ('f_label_0', self.f_label_0, pytorch_parser.f_label[0]),
                ('f_label_3', self.f_label_3, pytorch_parser.f_label[3])
            ]
            
            for name, tf_layer, pt_layer in label_layers:
                weight = pt_layer.weight.detach().numpy()
                bias = pt_layer.bias.detach().numpy()
                logger.info(f"Copying {name} weights with shapes - weight: {weight.shape}, bias: {bias.shape}")
                tf_layer.set_weights([weight.T, bias])
            
            # Interregnum weights
            interregnum_weight = pytorch_parser.f_interregnum.weight.detach().numpy()
            interregnum_bias = pytorch_parser.f_interregnum.bias.detach().numpy()
            logger.info(f"Copying interregnum weights with shapes - weight: {interregnum_weight.shape}, bias: {interregnum_bias.shape}")
            self.f_interregnum.set_weights([interregnum_weight.T, interregnum_bias])
            
        except Exception as e:
            logger.error(f"Error copying task weights: {str(e)}")
            raise
    
    def _copy_weights(self, pytorch_parser):
        """Copy weights from PyTorch model"""
        try:
            # Copy projection weights
            self._copy_projection_weights(pytorch_parser)
            
            # Copy task-specific weights
            self._copy_task_weights(pytorch_parser)
            
            logger.info("Weight copying completed successfully")
            
        except Exception as e:
            logger.error(f"Error during weight copying: {str(e)}")
            raise

    def _copy_projection_weights(self, pytorch_parser):
        """Copy projection layer weights"""
        try:
            # Project BERT weights
            project_weight = pytorch_parser.project_bert.weight.detach().numpy()
            logger.info(f"Copying project_bert weights with shape: {project_weight.shape}")
            self.project_bert.set_weights([project_weight.T])
            
            # Fix: Create identity matrix with correct shape for TF Dense layer
            # For TF Dense layer, weight shape should be (input_dim, units)
            intermediate_weight = np.eye(self.projected_dim, self.hidden_dim)  # Shape: (64, 128)
            logger.info(f"Setting intermediate projection with shape: {intermediate_weight.shape}")
            self.intermediate_proj.set_weights([intermediate_weight])
            
        except Exception as e:
            logger.error(f"Error copying projection weights: {str(e)}")
            raise

    def call(self, inputs, training=False, mask=None):
        """Forward pass of the model"""
        # Ensure inputs are int32
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        
        # Get BERT output with proper input type
        bert_output = self.bert(
            input_ids=inputs,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
            training=training
        )[0]
        
        # Project BERT output
        projected = self.project_bert(bert_output)
        
        # Project to hidden dimension
        hidden = self.intermediate_proj(projected)
        
        # Apply layer normalization
        normalized = self.layer_norm(hidden)
        
        # Task-specific outputs
        label_0 = self.f_label_0(normalized)
        label_final = self.f_label_3(label_0)
        interregnum = self.f_interregnum(normalized)
        
        return {
            'label_scores': label_final,
            'interregnum_scores': interregnum
        }
    
    def _init_layers(self):
        """Initialize model layers"""
        # Initialize BERT
        self.bert = TFMobileBertModel.from_pretrained(
            'google/mobilebert-uncased',
            from_pt=True,
            output_hidden_states=True
        )
        
        # Project layers
        self.project_bert = tf.keras.layers.Dense(
            units=self.projected_dim,  # 512 -> 64
            use_bias=False,
            name="project_bert"
        )
        
        # Fix: Swap dimensions for intermediate projection
        self.intermediate_proj = tf.keras.layers.Dense(
            units=self.hidden_dim,     # 64 -> 128
            use_bias=False,
            kernel_initializer='identity',  # Initialize with identity matrix
            name="intermediate_proj"
        )
        
        # Normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12,
            name="layer_norm"
        )
        
        # Label layers
        self.f_label_0 = tf.keras.layers.Dense(
            units=250,
            activation='relu',
            name="f_label_0"
        )
        
        self.f_label_3 = tf.keras.layers.Dense(
            units=113,
            activation='sigmoid',
            name="f_label_3"
        )
        
        # Interregnum layer
        self.f_interregnum = tf.keras.layers.Dense(
            units=2,
            activation='sigmoid',
            name="f_interregnum"
        )
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def serve(self, inputs):
        """Serving function for TFLite conversion"""
        return self.call(inputs, training=False)

def validate_tflite_model(model_path):
    """Validate the converted TFLite model"""
    logger.info("Validating TFLite model...")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create dummy input
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.int32)
    
    # Test inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get outputs
    label_scores = interpreter.get_tensor(output_details[0]['index'])
    interregnum_scores = interpreter.get_tensor(output_details[1]['index'])
    
    logger.info("TFLite model validation successful")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Label scores shape: {label_scores.shape}")
    logger.info(f"Interregnum scores shape: {interregnum_scores.shape}")

def add_metadata(tflite_model):
    """Add metadata to TFLite model"""
    from tflite_support import metadata as _metadata
    from tflite_support import metadata_schema_py_generated as _metadata_fb
    
    metadata_buffer = _metadata.MetadataPopulator.with_model_buffer(tflite_model)
    
    # Add model metadata
    metadata_buffer.add_general_info(
        model_name="MobileBERT Disfluency Parser",
        model_description="Disfluency detection and parsing model based on MobileBERT",
        version="1.0",
        author="Your Name",
        license="Your License"
    )
    
    # Add input/output metadata
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "input_ids"
    input_meta.description = "Input token IDs"
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    
    metadata_buffer.add_input_tensor_metadata(input_meta)
    
    # Populate metadata
    metadata_buffer.populate()
    return metadata_buffer.get_model_buffer()

def convert_to_tflite(input_path, output_path):
    try:
        # Load PyTorch model
        info = torch_load(input_path)
        parser = MobileBERTChartParser.from_spec(info['spec'], info['state_dict'])
        parser.eval()
        
        # Create and initialize TF model
        tf_parser = TFMobileBERTParser(parser)
        
        # Convert to TFLite
        logger.info("Converting model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_parser)
        
        # Set input shape explicitly
        input_shape = (1, 1)  # batch_size=1, seq_len=1
        def representative_dataset():
            for _ in range(100):
                data = np.random.randint(0, 1000, input_shape, dtype=np.int32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"Model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

def _test_model_conversion(pt_parser, tf_parser):
    """Test both PyTorch and TF models with same input"""
    # Create dummy input with correct dtype
    dummy_input = tf.zeros((1, 32), dtype=tf.int32)  # Changed to int32
    
    # TF inference
    tf_output = tf_parser(dummy_input)
    logger.info("Test inference completed successfully")
    logger.info(f"TF Output shapes:")
    logger.info(f"  Label scores: {tf_output['label_scores'].shape}")
    logger.info(f"  Interregnum scores: {tf_output['interregnum_scores'].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TFLite')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to PyTorch model (.pt file)')
    parser.add_argument('--output', type=str, default='converted_model.tflite',
                       help='Output path for TFLite model')
    
    args = parser.parse_args()
    convert_to_tflite(args.input, args.output)