import json
import argparse
import numpy as np
from main import run_train, make_hparams
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to the job dir')
parser.add_argument('--model-path', type=str, help='path to the model dir', default=None)
parser.add_argument('--eval-path', type=str, help='path to the eval dir', default=None)
parser.add_argument('--evalb-dir', type=str, help='path to the EVALB directory', default='./EVALB')
parser.add_argument('--override', type=str, help='JSON string to override config parameters', default=None)
parser.add_argument('--train-load-path', type=str, default=None,
                   help='Path to load saved model for continued training')
parser.add_argument('--run-name', type=str, help='Custom name for wandb run')
parser.add_argument('--is-experimental', action='store_true', 
                   help='Tag run as experimental')
parser.add_argument('--s3-bucket', type=str,
                   help='S3 bucket name for model storage')
parser.add_argument('--wandb-project', type=str, 
                   default='disfluency-detection-training',
                   help='WandB project name')
args = parser.parse_args()


def run_self_attentive_parser(run_config_filename, model_path=None, eval_out_path=None, evalb_dir=None, override_params=None):
    random_config = dict()
    random_config['results_path'] = eval_out_path
    random_config['model_path_base'] = model_path
    random_config['evalb_dir'] = evalb_dir
    random_config['numpy_seed'] = np.random.randint(1, 40000)
    with open(run_config_filename, 'r') as fp:
        random_config.update(json.loads(fp.read()))

    if override_params:
        override_dict = json.loads(override_params)
        random_config.update(override_dict)

    args = argparse.Namespace()
    vars(args).update(random_config)

    # Add wandb-related arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-project', type=str, 
                       default='disfluency-detection-training',
                       help='WandB project name')
    parser.add_argument('--run-name', type=str, 
                       help='Custom name for wandb run')
    parser.add_argument('--is-experimental', action='store_true', 
                       help='Tag run as experimental')
    parser.add_argument('--s3-bucket', type=str,
                       help='S3 bucket name for model storage')
    
    # Print configuration
    print("\nConfiguration:")
    print(f"WandB Project: {args.wandb_project}")
    print(f"Run Name: {args.run_name}")
    print(f"Is Experimental: {args.is_experimental}")
    print(f"S3 Bucket: {args.s3_bucket}")

    hparams = make_hparams()
    hparams.set_from_args(args)
    run_train(args, hparams)

if __name__ == '__main__':
	run_self_attentive_parser(
		args.config, 
		args.model_path, 
		args.eval_path, 
		args.evalb_dir,
		args.override
	)

