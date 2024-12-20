import argparse
import itertools
import os.path
import time
import sys
import torch
import torch.optim.lr_scheduler
import time

import numpy as np
import random

import evaluate
import trees
import vocabulary
import nkutil
import parse_nk
tokens = parse_nk
import evaluate_EDITED
from tqdm import tqdm
import tensorflow as tf
import wandb
import os
import boto3
from botocore.exceptions import NoCredentialsError
import json

def torch_load(load_path):
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def make_hparams():
    return nkutil.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        sentence_max_len=300,

        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3, # establishes a termination criterion

        partitioned=True,
        num_layers_position_only=0,

        num_layers=8,
        d_model=1024,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_tag_hidden=250,
        tag_loss_scale=5.0,

        attention_dropout=0.2,
        embedding_dropout=0.0,
        relu_dropout=0.1,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_chars_lstm=False,
        use_elmo=False,
        use_bert=False,
        use_bert_only=False,
        predict_tags=False,

        d_char_emb=32, # A larger value may be better for use_chars_lstm

        tag_emb_dropout=0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,
        elmo_dropout=0.5, # Note that this semi-stacks with morpho_emb_dropout!

        bert_model="bert-base-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        min_subbatch_size=4,
        max_subbatch_size=32,
        evalb_dir="EVALB",

        # Add more specific parameters for interregnum detection
        interregnum_loss_scale=3.0,
        use_interregnum_markers=True,
        interregnum_tags=['UH', 'UM', 'UHH', 'ERR'],
        reparandum_weight=0.6,
        interregnum_weight=0.4,

        # Add quantization parameters
        quantization=dict(
            enable=False,  # Disabled by default
            output_path="results/quantized_model.tflite",
            inference_type="int8",
            optimization_default=True
        ),
        s3_bucket="armel", 
        run_name="mobilebert_experiment"
    )

def upload_to_s3(file_path, bucket_name, s3_file_name):
    if not bucket_name:
        print("No S3 bucket specified in config. Skipping S3 upload.")
        return
        
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, bucket_name, s3_file_name)
        print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_file_name}")
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available for AWS S3.")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")

def quantize_and_save_model(parser, config, model_path=None):
    """
    Quantize and save model based on configuration parameters
    """
    if not config.get("quantization", {}).get("enable", False):
        return
        
    if not isinstance(parser, parse_nk.MobileBERTChartParser):
        print("Quantization is only supported for MobileBERT models. Skipping...")
        return
        
    try:
        import tensorflow as tf
        import numpy as np
        from transformers import TFMobileBertModel
        import signal
        from contextlib import contextmanager
        import time
        import os
        import wandb
    except ImportError as e:
        print(f"Error: Missing required package - {str(e)}")
        return

    @contextmanager
    def timeout(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Quantization timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    
    print("Converting and quantizing model...")
    
    class TFMobileBERTParser(tf.keras.Model):
        def __init__(self, pytorch_parser):
            super(TFMobileBERTParser, self).__init__()
            # Initialize BERT
            self.bert = TFMobileBertModel.from_pretrained(
                'google/mobilebert-uncased',
                from_pt=True,  # Enable PyTorch weight loading
                output_hidden_states=True
            )
            
            # Add parsing layers
            self.span_scorer = tf.keras.layers.Dense(
                units=pytorch_parser.span_attention.total_output_size,
                name="span_scorer"
            )
            self.edited_scorer = tf.keras.layers.Dense(
                units=1,
                name="edited_scorer"
            )
            self.filler_scorer = tf.keras.layers.Dense(
                units=1,
                name="filler_scorer"
            )
            
            # Copy weights from PyTorch model
            self._copy_weights(pytorch_parser)
            
        def _copy_weights(self, pytorch_parser):
            # Copy span scorer weights
            span_weights = pytorch_parser.span_attention.get_weights()
            self.span_scorer.set_weights([
                tf.convert_to_tensor(span_weights[0].numpy()),
                tf.convert_to_tensor(span_weights[1].numpy())
            ])
            
            # Copy edited scorer weights
            edited_weights = pytorch_parser.edited_scorer.get_weights()
            self.edited_scorer.set_weights([
                tf.convert_to_tensor(edited_weights[0].numpy()),
                tf.convert_to_tensor(edited_weights[1].numpy())
            ])
            
            # Copy filler scorer weights if available
            if hasattr(pytorch_parser, 'filler_scorer'):
                filler_weights = pytorch_parser.filler_scorer.get_weights()
                self.filler_scorer.set_weights([
                    tf.convert_to_tensor(filler_weights[0].numpy()),
                    tf.convert_to_tensor(filler_weights[1].numpy())
                ])
        
        def call(self, inputs, training=False):
            # Get BERT embeddings
            hidden_states = self.bert(inputs, training=training).last_hidden_state
            
            # Calculate span scores
            span_scores = self.span_scorer(hidden_states)
            
            # Calculate edited scores
            edited_scores = self.edited_scorer(hidden_states)
            
            # Calculate filler scores
            filler_scores = self.filler_scorer(hidden_states)
            
            return {
                'span_scores': span_scores,
                'edited_scores': edited_scores,
                'filler_scores': filler_scores
            }
    
    # Create model with parsing functionality
    tf_model = TFMobileBERTParser(parser)
    
    # Create sample input and run model once to build
    seq_length = 512
    sample_input = tf.random.uniform((1, seq_length), minval=0, maxval=30522, dtype=tf.int32)
    
    print("Building model...")
    _ = tf_model(sample_input)
    print(f"Model parameters: {tf_model.count_params():,}")
    
    print("Saving TF model...")
    tf.saved_model.save(tf_model, "temp_saved_model")
    
    print("Converting to TFLite (this might take a few minutes)...")
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_saved_model")
    
    # Use dynamic range quantization instead of full integer
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    try:
        with timeout(300):  # 5 minute timeout
            print("Starting quantization...")
            start_time = time.time()
            tflite_model = converter.convert()
            conversion_time = time.time() - start_time
            print(f"Quantization completed in {conversion_time:.1f} seconds")
            
            # Save model
            output_path = config.get("output_path", "quantized_model.tflite")
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Report sizes
            tflite_size = os.path.getsize(output_path) / (1024 * 1024)
            pt_model_path = config.get("model_path", model_path)
            if pt_model_path is None:
                raise ValueError("No model path provided for size comparison")
                
            original_size = os.path.getsize(pt_model_path) / (1024 * 1024)
            
            print(f"\nModel Size Comparison:")
            print(f"Original PyTorch model: {original_size:.2f} MB")
            print(f"Quantized TFLite model: {tflite_size:.2f} MB")
            print(f"Compression ratio: {original_size/tflite_size:.2f}x")
            
            # Create a new artifact for the models
            artifact = wandb.Artifact('model_files', type='model')
            
            # Add both PyTorch and TFLite models to the artifact
            artifact.add_file(pt_model_path, name='pytorch_model.pt')
            artifact.add_file(output_path, name='quantized_model.tflite')
            
            # Log metrics about model sizes
            wandb.log({
                "model/pytorch_size_mb": original_size,
                "model/tflite_size_mb": tflite_size,
                "model/compression_ratio": original_size/tflite_size,
                "model/quantization_time": conversion_time
            })
            
            # Log the artifact
            wandb.log_artifact(artifact)
            
            # Use the s3_bucket from config
            s3_bucket = config.get('s3_bucket', '')
            if s3_bucket:
                print(f"\nUploading models to S3 bucket: {s3_bucket}")
                run_name = config.get('run_name', 'default_run')
                s3_pt_path = f"{run_name}/pytorch_model.pt"
                s3_tflite_path = f"{run_name}/quantized_model.tflite"
                
                upload_to_s3(pt_model_path, s3_bucket, s3_pt_path)
                upload_to_s3(output_path, s3_bucket, s3_tflite_path)
            else:
                print("\nNo S3 bucket specified in config. Skipping S3 upload.")
            
    except TimeoutError:
        print("Quantization timed out after 5 minutes!")
        return
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # Cleanup
        import shutil
        shutil.rmtree("temp_saved_model", ignore_errors=True)

def run_train(args, hparams):
    # Add validation for interregnum-specific parameters
    if hparams.use_interregnum_markers:
        if not hasattr(hparams, 'interregnum_tags') or not hparams.interregnum_tags:
            raise ValueError("interregnum_tags must be specified when use_interregnum_markers is True")
        if not hasattr(hparams, 'interregnum_weight') or not hparams.interregnum_weight:
            raise ValueError("interregnum_weight must be specified when use_interregnum_markers is True")
        if not hasattr(hparams, 'reparandum_weight') or not hparams.reparandum_weight:
            raise ValueError("reparandum_weight must be specified when use_interregnum_markers is True")

    # Initialize wandb with automatic config capture
    wandb_config = {
        # First, capture all hparams
        **vars(hparams),
        # Then capture all command line args
        **vars(args),
        # Add any additional fixed parameters
        "architecture": "NKChartParser",
        "dataset": "swbd"
    }

    # Initialize wandb with the config
    wandb.init(
        project="disfluency-detection-training",
        config=wandb_config,
        name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",  # Add a unique name for each run
        tags=[
            "parser",
            "disfluency",
            "bert" if hparams.use_bert else "no-bert",
            f"layers_{hparams.num_layers}",
        ]
    )

    # Print captured config for verification
    print("Wandb config:", wandb_config)

    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    # Load training data
    print("Loading gold training trees from {}...".format(args.gold_train_path)) 
    print("Loading silver training trees from {}...".format(args.silver_train_path))
    if hparams.predict_tags and args.train_path.endswith('10way.clean'):
        print("WARNING: The data distributed with this repository contains "
              "predicted part-of-speech tags only (not gold tags!) We do not "
              "recommend enabling predict_tags in this configuration.")
    gold_train_treebank = trees.load_trees(args.gold_train_path)
    silver_train_treebank = trees.load_trees(args.silver_train_path)
    
    # Add validation for interregnum labels
    def validate_interregnum_labels(treebank, hparams):
        interregnum_count = 0
        for tree in treebank:
            # Get all leaves from the tree
            leaves = list(tree.leaves())
            for leaf in leaves:
                # Check if the leaf's tag is in our interregnum tags list
                if leaf.tag in hparams.interregnum_tags:
                    interregnum_count += 1
        
        print(f"Found {interregnum_count} interregnum markers in dataset")
        return interregnum_count
    
    validate_interregnum_labels(gold_train_treebank, hparams)
    
    # Load development data
    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print(f"Number of dev examples: {len(dev_treebank)}")
    
    if hparams.max_len_dev > 0:
        dev_treebank = [tree for tree in dev_treebank if len(list(tree.leaves())) <= hparams.max_len_dev]
    
    if args.silver_weight == 0:
        # Use only gold data
        new_batch_size = args.batch_size
        silver_batch_size = 0
        silver_train_treebank = []
        silver_train_parse = []
    else:
        # Use both gold and silver data
        new_batch_size = int(args.batch_size * ((10-args.silver_weight)/10)) 
        silver_batch_size = int(args.batch_size - new_batch_size)
        
    if hparams.max_len_train > 0:
        gold_train_treebank = [tree for tree in gold_train_treebank if len(list(tree.leaves())) <= hparams.max_len_train]
        silver_train_treebank = [tree for tree in silver_train_treebank if len(list(tree.leaves())) <= hparams.max_len_train]
        if int(silver_batch_size*(len(gold_train_treebank)/new_batch_size))+1 < len(silver_train_treebank):
            silver_train_treebank = silver_train_treebank[:int(silver_batch_size*(len(gold_train_treebank)/new_batch_size))+1]

    print("=== Training Data Statistics ===")
    print(f"Number of gold training examples: {len(gold_train_treebank)}")
    print(f"Batch size: {new_batch_size}")
    print(f"Silver weight: {args.silver_weight}")
    print("===============================")

    # Add validation for batch size
    if new_batch_size <= 0:
        print("ERROR: Invalid batch size calculated!")
        print(f"args.batch_size: {args.batch_size}")
        print(f"args.silver_weight: {args.silver_weight}")
        raise ValueError(f"Calculated batch size ({new_batch_size}) must be positive!")

    print("Processing gold trees for training...")
    gold_train_parse = [tree.convert() for tree in gold_train_treebank]

    print("Processing silver trees for training...")
    silver_train_parse = [tree.convert() for tree in silver_train_treebank]

    # Now we can check batch size against processed data
    if new_batch_size > len(gold_train_parse):
        print("WARNING: Batch size is larger than dataset!")
        print(f"Reducing batch size from {new_batch_size} to {len(gold_train_parse)}")
        new_batch_size = len(gold_train_parse)

    train_parse = gold_train_parse + silver_train_parse

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(tokens.START)
    tag_vocab.index(tokens.STOP)
    tag_vocab.index(tokens.TAG_UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(tokens.START)
    word_vocab.index(tokens.STOP)
    word_vocab.index(tokens.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    char_set = set()

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                char_set |= set(node.word)

    char_vocab = vocabulary.Vocabulary()

    # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
    highest_codepoint = max(ord(char) for char in char_set)
    if highest_codepoint < 512:
        if highest_codepoint < 256:
            highest_codepoint = 256
        else:
            highest_codepoint = 512

        # This also takes care of constants like tokens.CHAR_PAD
        for codepoint in range(highest_codepoint):
            char_index = char_vocab.index(chr(codepoint))
            assert char_index == codepoint
    else:
        char_vocab.index(tokens.CHAR_UNK)
        char_vocab.index(tokens.CHAR_START_SENTENCE)
        char_vocab.index(tokens.CHAR_START_WORD)
        char_vocab.index(tokens.CHAR_STOP_WORD)
        char_vocab.index(tokens.CHAR_STOP_SENTENCE)
        for char in sorted(char_set):
            char_vocab.index(char)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {tokens.START, tokens.STOP, tokens.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")

    if hasattr(args, 'train_load_path') and args.train_load_path is not None:
        print(f"Loading parameters from {args.train_load_path}")
        info = torch_load(args.train_load_path)
        if 'mobilebert' in hparams.bert_model.lower():
            parser = parse_nk.MobileBERTChartParser.from_spec(info['spec'], info['state_dict'])
        else:
            parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
    else:
        if 'mobilebert' in hparams.bert_model.lower():
            parser = parse_nk.MobileBERTChartParser(
                tag_vocab,
                word_vocab,
                label_vocab,
                char_vocab,
                hparams,
            )
        else:
            parser = parse_nk.NKChartParser(
                tag_vocab,
                word_vocab,
                label_vocab,
                char_vocab,
                hparams,
            )

    # Print model size information after parser is created
    print("\nModel size statistics:")
    total_params = sum(p.numel() for p in parser.parameters())
    trainable_params = sum(p.numel() for p in parser.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print size of each major component
    for name, module in parser.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters")

    print("Initializing optimizer...")
    trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    if hasattr(args, 'train_load_path') and args.train_load_path is not None:
        trainer.load_state_dict(info['trainer'])

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    assert hparams.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, mode='max', factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience, verbose=False)
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm

    print("Training...")
    total_processed = 0
    dev_elapsed_time = 0  # Track total time spent in evaluation
    start_time = time.time()
    check_every = len(gold_train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_processed = 0
    best_dev_efscore = None
    dev_efscore = None
    start_epoch = 0

    start_time = time.time()

    def check_dev(train_parse):
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_dev_processed
        nonlocal best_dev_efscore
        nonlocal dev_elapsed_time

        print("\nEvaluating on development set...")
        dev_start_time = time.time()

        dev_predicted = []
        # Create progress bar for dev set evaluation
        dev_pbar = tqdm(
            total=len(dev_treebank),
            desc="Dev evaluation",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                      '[{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100
        )

        for dev_start_index in range(0, len(dev_treebank), args.eval_batch_size):
            subbatch_trees = dev_treebank[dev_start_index:dev_start_index+args.eval_batch_size]
            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]
            
            # Update progress bar description with current batch
            dev_pbar.set_description(f"Dev evaluation (processing {dev_start_index+1}-{min(dev_start_index+args.eval_batch_size, len(dev_treebank))} of {len(dev_treebank)})")
            
            predicted, _ = parser.parse_batch(subbatch_sentences)
            del _
            dev_predicted.extend([p.convert() for p in predicted])
            
            # Update progress
            dev_pbar.update(len(subbatch_trees))

        dev_pbar.close()

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)
        dev_efscore = evaluate_EDITED.Evaluate(
            dev_treebank, 
            dev_predicted,
            evaluate_reparandum=True,
            evaluate_interregnum=hparams.use_interregnum_markers
        )

        metrics = {
            # Parsing metrics
            "dev/parsing/fscore": float(dev_fscore.fscore),
            "dev/parsing/recall": float(dev_fscore.recall),
            "dev/parsing/precision": float(dev_fscore.precision),
            
            # Always track reparandum score
            "dev/reparandum/score": float(dev_efscore.reparandum_score),
        }

        if hparams.use_interregnum_markers:
            # Add interregnum metrics when enabled
            metrics.update({
                "dev/interregnum/score": float(dev_efscore.interregnum_score),
                "dev/combined/score": float(dev_efscore.combined_score)
            })

        wandb.log(metrics)

        if hparams.use_interregnum_markers:
            print(
                "dev-fscore {} "
                "reparandum-score {:.4f} "
                "interregnum-score {:.4f} "
                "combined-score {:.4f} "
                "dev-elapsed {} "
                "total-elapsed {}".format(
                    dev_fscore,
                    dev_efscore.reparandum_score,
                    dev_efscore.interregnum_score,
                    dev_efscore.combined_score,
                    format_elapsed(dev_start_time),
                    format_elapsed(start_time)),
                flush=True
            )
        else:
            print(
                "dev-fscore {} "
                "reparandum-score {:.4f} "
                "dev-elapsed {} "
                "total-elapsed {}".format(
                    dev_fscore,
                    dev_efscore.reparandum_score,
                    format_elapsed(dev_start_time),
                    format_elapsed(start_time)),
                flush=True
            )

        # Use combined_score for model selection since we're evaluating both tasks
        if dev_efscore.combined_score > best_dev_fscore:
            if best_dev_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print(" Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_efscore = dev_efscore
            best_dev_fscore = dev_efscore.combined_score
            
            # Format model path based on enabled tasks
            if hparams.use_interregnum_markers:
                best_dev_model_path = "{}_dev={:.4f}_rep={:.4f}_int={:.4f}".format(
                    args.model_path_base, 
                    dev_efscore.combined_score,
                    dev_efscore.reparandum_score,
                    dev_efscore.interregnum_score
                )
            else:
                best_dev_model_path = "{}_dev={:.4f}_rep={:.4f}".format(
                    args.model_path_base, 
                    dev_efscore.fscore,
                    dev_efscore.reparandum_score
                )
            
            best_dev_processed = total_processed

            print(" Saving new best model to {}...".format(best_dev_model_path))
            
            # Prepare metrics based on enabled tasks
            metrics = {
                'fscore': dev_efscore.fscore,
                'reparandum_score': dev_efscore.reparandum_score,
            }
            
            if hparams.use_interregnum_markers:
                metrics.update({
                    'combined_score': dev_efscore.combined_score,
                    'interregnum_score': dev_efscore.interregnum_score,
                })
            
            # Prepare task config based on enabled tasks
            task_config = {
                'use_interregnum_markers': hparams.use_interregnum_markers,
                'reparandum_weight': hparams.reparandum_weight,
            }
            
            if hparams.use_interregnum_markers:
                task_config.update({
                    'interregnum_tags': hparams.interregnum_tags,
                    'interregnum_weight': hparams.interregnum_weight
                })

            torch.save({
                'spec': parser.spec,
                'state_dict': parser.state_dict(),
                'trainer': trainer.state_dict(),
                'hparams': hparams.to_dict(),
                'metrics': metrics,
                'task_config': task_config
            }, best_dev_model_path + ".pt")

            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(best_dev_model_path + ".pt")
            wandb.log_artifact(artifact)

        upload_to_s3(  best_dev_model_path + ".pt", hparams.s3_bucket, f"{hparams.run_name}/best_pytorch_model/{best_dev_model_path}.pt")

        return dev_efscore

    def check_hurdle(epoch, hurdle):
        if hparams.use_interregnum_markers:
            # Check both tasks when using interregnum detection
            if (dev_efscore.reparandum_score < hurdle or 
                dev_efscore.interregnum_score < hurdle):
                message = (
                    f"FAILURE: Epoch {epoch} hurdle failed, stopping now!\n"
                    f"reparandum_score = {dev_efscore.reparandum_score} < hurdle = {hurdle}\n"
                    f"interregnum_score = {dev_efscore.interregnum_score} < hurdle = {hurdle}"
                )
                print(message, flush=True)
                if args.results_path:
                    print(message, file=open(args.results_path, 'w'), flush=True)
                sys.exit(message)
        else:
            # Only check reparandum score
            if dev_efscore.reparandum_score < hurdle:
                message = (
                    f"FAILURE: Epoch {epoch} hurdle failed, stopping now!\n"
                    f"reparandum_score = {dev_efscore.reparandum_score} < hurdle = {hurdle}"
                )
                print(message, flush=True)
                if args.results_path:
                    print(message, file=open(args.results_path, 'w'), flush=True)
                sys.exit(message)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nStarting Epoch {epoch}")
        current_processed = 0
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        train_batches = [(start, start + new_batch_size) 
                        for start in range(0, len(gold_train_parse), new_batch_size)]
        print(f"Number of batches: {len(train_batches)}")
        
        # Initialize progress bar with more detailed format
        pbar = tqdm(
            total=len(train_batches),
            desc=f"Epoch {epoch}/{args.epochs}",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=100
        )
        
        for batch_num, (start, end) in enumerate(train_batches):
            batch_trees = gold_train_parse[start:end]
            
            # Update description instead of printing
            pbar.set_description(f"Epoch {epoch}/{args.epochs} (batch {batch_num + 1}/{len(train_batches)}, {len(batch_trees)} examples)")
            
            trainer.zero_grad()
            schedule_lr(total_processed // new_batch_size)

            batch_loss_value = 0.0
            batch_trees.sort(key=lambda tree: len(list(tree.leaves())), reverse=True)
            batch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in batch_trees]

            # Group sentences of similar lengths together (bucketing)
            length_groups = {}
            for idx, sent in enumerate(batch_sentences):
                length_key = len(sent)  # Group only exactly matching lengths
                if length_key not in length_groups:
                    length_groups[length_key] = []
                length_groups[length_key].append((idx, sent))

            current_lr = trainer.param_groups[0]['lr']
            wandb.log({
                "train/step": total_processed // new_batch_size,
                "train/learning_rate": current_lr,
                "train/batch_size": len(batch_trees)
            })
            try:
                total_loss = 0
                # Process each length group separately
                for group_key in sorted(length_groups.keys(), reverse=True):
                    group_indices, group_sentences = zip(*length_groups[group_key])
                    group_trees = [batch_trees[idx] for idx in group_indices]
                    
                    # Process in smaller chunks if group is too large
                    chunk_size = min(len(group_sentences), hparams.min_subbatch_size)
                    for i in range(0, len(group_sentences), chunk_size):
                        chunk_sentences = list(group_sentences[i:i + chunk_size])
                        chunk_trees = group_trees[i:i + chunk_size]
                        
                        # Get losses based on what we're training
                        if hparams.use_interregnum_markers:
                            # Get separate losses for both tasks
                            loss, tag_loss = parser.parse_batch(
                                chunk_sentences, 
                                chunk_trees,
                                compute_reparandum=True,
                                compute_interregnum=True
                            )
                        else:
                            # Only compute reparandum loss
                            loss, tag_loss = parser.parse_batch(
                                chunk_sentences, 
                                chunk_trees,
                                compute_reparandum=True,
                                compute_interregnum=False
                            )
                        
                        if tag_loss is not None:
                            loss = loss + tag_loss

                        loss_value = float(loss.data.cpu().numpy())
                        batch_loss_value += loss_value
                        if loss_value > 0:
                            loss.backward()
                        del loss
                        if tag_loss is not None:
                            del tag_loss

                        # Increment total_processed only once per example
                        total_processed += len(chunk_trees)
                        current_processed += len(chunk_trees)

            except RuntimeError as e:
                print(f"Error in batch starting at index {start}")
                print(f"Batch size: {len(batch_sentences)}")
                print(f"Sentence lengths: {[len(s) for s in batch_sentences]}")
                raise e

             # Calculate grad_norm after all backward passes
            grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)
                
            # Log metrics after grad_norm is calculated
            wandb.log({
                "train/step": total_processed // new_batch_size,
                "train/learning_rate": current_lr,
                "train/batch_size": len(batch_trees),
                "train/step_loss": batch_loss_value,
                "train/grad_norm": grad_norm
            })
            trainer.step()
            
            # Add batch loss to epoch loss
            epoch_loss += batch_loss_value
            
            # Update progress bar with loss information
            pbar.set_postfix({
                'loss': f'{batch_loss_value:.4f}',
                'avg_loss': f'{epoch_loss/(batch_num + 1):.4f}'
            }, refresh=True)
            pbar.update(1)
        
        pbar.close()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_batches)
        
        # Log epoch-level metrics with task-specific information
        metrics = {
            "train/epoch": epoch,
            "train/final_epoch_loss": avg_epoch_loss,
            "train/epoch_time": time.time() - epoch_start_time,
            "train/num_batches": len(train_batches)
        }
        
        if hparams.use_interregnum_markers:
            metrics.update({
                "train/reparandum_weight": hparams.reparandum_weight,
                "train/interregnum_weight": hparams.interregnum_weight
            })
            
        wandb.log(metrics)

        # Track evaluation time
        dev_start = time.time()
        dev_efscore = check_dev(train_parse)
        dev_elapsed_time += time.time() - dev_start
        
        assert dev_efscore, "dev_efscore is zero/empty. Check evaluation logic."
        print ("epoch {:,} " "total-processed {} " "current-processed {}" .format(epoch, total_processed, current_processed))
        if epoch == 1:
            check_hurdle(epoch, args.epoch1_hurdle)
        elif epoch == 10:
            check_hurdle(epoch, args.epoch10_hurdle)
            
        
        # adjust learning rate at the end of an epoch
        if (total_processed // new_batch_size + 1) > hparams.learning_rate_warmup_steps:
            scheduler.step(dev_efscore.combined_score) 
            if (total_processed - best_dev_processed) > ((hparams.step_decay_patience + 1) * hparams.max_consecutive_decays * len(gold_train_parse)):
                print("Terminating due to lack of improvement in dev fscore.")
                break

    assert best_dev_efscore, "best_dev_efscore not set; did you train for at least 1 epoch?"
    if args.results_path:
        outf = open(args.results_path, 'w')
        print(best_dev_efscore.table(), file=outf, flush=True)
    else:
        print(best_dev_efscore.table(), flush=True)
        
    # After training loop ends
    if best_dev_model_path and hasattr(hparams, 'quantization'):
        print("\nTraining completed. Starting final model quantization...")
        quantize_config = hparams.to_dict()
        quantize_config['quantization'] = {
            'enable': True,
            'output_path': best_dev_model_path + ".tflite",
            'inference_type': 'int8',
            'optimization_default': True
        }
        quantize_and_save_model(parser, quantize_config, best_dev_model_path + ".pt")
        
    # Close wandb run when training completes
    wandb.finish()

def run_test(args):
    # Initialize wandb in test mode
    wandb.init(
        project="disfluency-detection-training",
        job_type="evaluation"
    )

    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

    print("Parsing test sentences...")
    start_time = time.time()

    test_predicted = []
    for start_index in range(0, len(test_treebank), args.eval_batch_size):
        subbatch_trees = test_treebank[start_index:start_index+args.eval_batch_size]
        subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]
        predicted, _ = parser.parse_batch(subbatch_sentences)
        del _
        test_predicted.extend([p.convert() for p in predicted])

    # The tree loader does some preprocessing to the trees (e.g. stripping TOP
    # symbols or SPMRL morphological features). We compare with the input file
    # directly to be extra careful about not corrupting the evaluation. We also
    # allow specifying a separate "raw" file for the gold trees: the inputs to
    # our parser have traces removed and may have predicted tags substituted,
    # and we may wish to compare against the raw gold trees to make sure we
    # haven't made a mistake. As far as we can tell all of these variations give
    # equivalent results.
    ref_gold_path = args.test_path
    if args.test_path_raw is not None:
        print("Comparing with raw trees from", args.test_path_raw)
        ref_gold_path = args.test_path_raw

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted, ref_gold_path=ref_gold_path)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

    # Log test metrics
    wandb.log({
        "test/fscore": test_fscore,
        "test/elapsed_time": time.time() - start_time
    })

    wandb.finish()

#%%
def run_ensemble(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    parsers = []
    for model_path_base in args.model_path_base:
        print("Loading model from {}...".format(model_path_base))
        assert model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

        info = torch_load(model_path_base)
        assert 'hparams' in info['spec'], "Older savefiles not supported"
        parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
        parsers.append(parser)

    # Ensure that label scores charts produced by the models can be combined
    # using simple averaging
    ref_label_vocab = parsers[0].label_vocab
    for parser in parsers:
        assert parser.label_vocab.indices == ref_label_vocab.indices

    print("Parsing test sentences...")
    start_time = time.time()

    test_predicted = []
    # Ensemble by averaging label score charts from different models
    # We did not observe any benefits to doing weighted averaging, probably
    # because all our parsers output label scores of around the same magnitude
    for start_index in range(0, len(test_treebank), args.eval_batch_size):
        subbatch_trees = test_treebank[start_index:start_index+args.eval_batch_size]
        subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

        chart_lists = []
        for parser in parsers:
            charts = parser.parse_batch(subbatch_sentences, return_label_scores_charts=True)
            chart_lists.append(charts)

        subbatch_charts = [np.mean(list(sentence_charts), 0) for sentence_charts in zip(*chart_lists)]
        predicted, _ = parsers[0].decode_from_chart_batch(subbatch_sentences, subbatch_charts)
        del _
        test_predicted.extend([p.convert() for p in predicted])

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted, ref_gold_path=args.test_path)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

#%%

def run_parse(args):
    if args.output_path != '-' and os.path.exists(args.output_path):
        print("Error: output file already exists:", args.output_path)
        return

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    
    # Check if the model is MobileBERT based on the spec
    is_mobilebert = 'mobilebert' in info['spec']['hparams'].get('bert_model', '').lower()
    
    if is_mobilebert:
        print("Loading MobileBERT model...")
        parser = parse_nk.MobileBERTChartParser.from_spec(info['spec'], info['state_dict'])
    else:
        print("Loading BERT model...")
        parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

    print("Parsing sentences...")
    with open(args.input_path) as input_file:
        sentences = input_file.readlines()
    sentences = [sentence.strip().split() for sentence in sentences]  # Added strip() to handle newlines

    # Tags are not available when parsing from raw text, so use a dummy tag
    if 'UNK' in parser.tag_vocab.indices:
        dummy_tag = 'UNK'
    else:
        dummy_tag = parser.tag_vocab.value(0)

    start_time = time.time()

    all_predicted = []
    # Add progress bar for parsing
    total_batches = (len(sentences) + args.eval_batch_size - 1) // args.eval_batch_size
    with tqdm(total=total_batches, desc="Parsing progress") as pbar:
        for start_index in range(0, len(sentences), args.eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+args.eval_batch_size]
            
            try:
                subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
                predicted, _ = parser.parse_batch(subbatch_sentences)
                del _
                if args.output_path == '-':
                    for p in predicted:
                        print(p.convert().linearize())
                else:
                    all_predicted.extend([p.convert() for p in predicted])
            except Exception as e:
                print(f"Error processing batch starting at index {start_index}: {str(e)}")
                continue
            
            pbar.update(1)

    if args.output_path != '-':
        with open(args.output_path, 'w') as output_file:
            for tree in all_predicted:
                output_file.write("{}\n".format(tree.linearize()))
        print("Output written to:", args.output_path)
        print(f"Total sentences processed: {len(all_predicted)}")
        print(f"Total time elapsed: {format_elapsed(start_time)}")

def run_viz(args):
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    print("Loading test trees from {}...".format(args.viz_path))
    viz_treebank = trees.load_trees(args.viz_path)
    print("Loaded {:,} test examples.".format(len(viz_treebank)))

    print("Loading model from {}...".format(args.model_path_base))

    info = torch_load(args.model_path_base)

    assert 'hparams' in info['spec'], "Only self-attentive models are supported"
    parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

    from viz import viz_attention

    stowed_values = {}
    orig_multihead_forward = parse_nk.MultiHeadAttention.forward
    def wrapped_multihead_forward(self, inp, batch_idxs, **kwargs):
        res, attns = orig_multihead_forward(self, inp, batch_idxs, **kwargs)
        stowed_values[f'attns{stowed_values["stack"]}'] = attns.cpu().data.numpy()
        stowed_values['stack'] += 1
        return res, attns

    parse_nk.MultiHeadAttention.forward = wrapped_multihead_forward

    # Select the sentences we will actually be visualizing
    max_len_viz = 15
    if max_len_viz > 0:
        viz_treebank = [tree for tree in viz_treebank if len(list(tree.leaves())) <= max_len_viz]
    viz_treebank = viz_treebank[:1]

    print("Parsing viz sentences...")

    for start_index in range(0, len(viz_treebank), args.eval_batch_size):
        subbatch_trees = viz_treebank[start_index:start_index+args.eval_batch_size]
        subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]
        stowed_values = dict(stack=0)
        predicted, _ = parser.parse_batch(subbatch_sentences)
        del _
        predicted = [p.convert() for p in predicted]
        stowed_values['predicted'] = predicted

        for snum, sentence in enumerate(subbatch_sentences):
            sentence_words = [tokens.START] + [x[1] for x in sentence] + [tokens.STOP]

            for stacknum in range(stowed_values['stack']):
                attns_padded = stowed_values[f'attns{stacknum}']
                attns = attns_padded[snum::len(subbatch_sentences), :len(sentence_words), :len(sentence_words)]
                viz_attention(sentence_words, attns)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--gold-train-path", default="swbd-data/autopos-nopunct-nopw/train.txt")
    subparser.add_argument("--silver-train-path", default="swbd-data/autopos-nopunct-nopw/train.txt")
    subparser.add_argument("--dev-path", default="swbd-data/autopos-nopunct-nopw/dev.txt")
    subparser.add_argument("--batch-size", type=int, default=250)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=100)
    #subparser.add_argument("--min-subbatch-size", type=int, default=4)
    #subparser.add_argument("--max-subbatch-size", type=int, default=32)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--epoch1-hurdle", default=0.5, type=float, help="Stop training if epoch 1 efscore less than this value")
    subparser.add_argument("--epoch10-hurdle", default=0.75, type=float, help="Stop training if epoch 10 efscore less than this value")
    subparser.add_argument("--results-path", default=None)
    subparser.add_argument("--silver-weight", default=4, type=int, help="Weights on using silver parse trees in each mini-batch") 
    subparser.add_argument("--train-load-path", required=True)
    #subparser.add_argument("--interregnum-weight", type=float, default=0.4, help="Weight for interregnum detection loss")
    #subparser.add_argument("--reparandum-weight", type=float, default=0.6, help="Weight for reparandum detection loss")
    subparser.add_argument("--s3-bucket", type=str, help="S3 bucket name for storing models")
    subparser.add_argument("--config-path", type=str, help="Path to JSON config file")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="swbd-data/autopos-nopunct-nopw/test.tx")
    subparser.add_argument("--test-path-raw", type=str)
    subparser.add_argument("--eval-batch-size", type=int, default=100)

    subparser = subparsers.add_parser("ensemble")
    subparser.set_defaults(callback=run_ensemble)
    subparser.add_argument("--model-path-base", nargs='+', required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="swbd-data/autopos-nopunct-nopw/test.tx")
    subparser.add_argument("--eval-batch-size", type=int, default=100)

    subparser = subparsers.add_parser("parse")
    subparser.set_defaults(callback=run_parse)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--input-path", type=str, required=True)
    subparser.add_argument("--output-path", type=str, default="-")
    subparser.add_argument("--eval-batch-size", type=int, default=100)

    subparser = subparsers.add_parser("viz")
    subparser.set_defaults(callback=run_viz)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--viz-path", default="data/22.auto.clean")
    subparser.add_argument("--eval-batch-size", type=int, default=100)    

    args = parser.parse_args()
    if hasattr(args, 'config_path') and args.config_path:
        config = load_config(args.config_path)
        args.callback(args, config)
    else:
        args.callback(args)

# %%
if __name__ == "__main__":
    main()
