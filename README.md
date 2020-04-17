Joint Disfluency Detection and Constituency Parsing
------------------------------------------------------------
A joint disfluency detection and constituency parsing model for transcribed speech based on [Neural Constituency Parsing of Speech Transcripts](https://www.aclweb.org/anthology/N19-1282) from NAACL 2019, with additional changes (e.g. self-training and ensembling) described in [Improving Disfluency Detection by Self-Training a Self-Attentive Model](https://arxiv.org/pdf/2004.05323.pdf).

## Contents
1. [Task](#task)
2. [Requirements for Training](#requirement)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Training Instructions](#training)
6. [Citation](#citation)
7. [Contact](#contact)
8. [Credits](#credits)

### Task
Disfluency refers to any interruptions in the normal flow of speech, including filled pauses (*um*, *uh*), repetitions (*they're ... they're here*), corrections (*to Sydney ... no to Melbourne*), false starts (*we would like ... let's go*), parenthetical asides (*you know*, *I mean*), partial words (*wou-*, *oper-*) and interjections (*well*, *like*). One type of disfluency which is especially problematic for conventional syntactic parsers are speech repairs. A speech repair consists of three main parts; the *reparandum*, the *interregnum* and the *repair*. As illustrated in the example below, the reparandum *we don't* is the part of the utterance that is replaced or repaired, the interregnum *uh I mean* (which consists of a filled pause *uh* and a discourse marker *I mean*) is an optional part of the disfluency, and the repair *a lot of states don't* replaces the reparandum. The fluent version is obtained by removing the reparandum and the interregnum.

<p align="center">
  <img src="img/flat-ex.jpg" width=380 height=120>
</p>

We train a joint disfluency detection and constituency parsing model of transcribed speech on the Penn Treebank-3 Switchboard corpus. Since the Switchboard trees include both syntactic constituency nodes and disfluency nodes, training a parser to predict the Switchboard trees can be regarded as multi-task learning (where the tasks are syntactic parsing and identifying disfluencies). In the Switchboard treebank corpus the *reparanda*, *filled pauses* and *discourse markers* are dominated by *EDITED*, *INTJ* and *PRN* nodes, respectively. Filled pauses and discourse markers belong to a finite set of words and phrases, so INTJ and PRN nodes are trivial to detect. Detecting EDITED nodes, however, is challenging and is the main focus of disfluency detection models.

<p align="center">
  <img src="img/tree-ex.jpg" width=600 height=300>
</p>

### Requirements for Training
* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.1, 1.0/1.1, or any compatible version.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.
* [AllenNLP](http://allennlp.org/) 0.7.0 or any compatible version (only required when using ELMo word representations)
* [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4.0 or any compatible version (only required when using BERT word representations)

### Preparation
```
$ git clone https://github.com/pariajm/joint-disfluency-detector-and-parser
$ cd joint-disfluency-detector-and-parser/EVALB
$ make evalb
$ cd .. 
```
To use ELMo embeddings, follow the additional instructions given below:
```
$ mkdir data
$ cd data
$ wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
$ wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
$ cd ..
```
### Pre-trained Models (PyTorch)
The following pre-trained models are available for download:
* [`swb_fisher_bert_Edev.0.9078.pt`](https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swb_fisher_bert_Edev.0.9078.pt): Our best model self-trained on the Switchboard gold parse trees and Fisher silver parse trees with BERT-base-uncased word representations (EDITED word f-score=92.4%).
* [`swb_bert_Edev.0.8922.pt`](https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swb_bert_Edev.0.8922.pt): Our best model trained on the Switchboard gold parse trees with BERT-base-uncased word representations (EDITED word f-score=90.9%).
* [`swb_elmo_tree_transformation_Edev.0.8838.pt`](https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swb_elmo_tree_transformation_Edev.0.8838.pt): Our best model trained on the tree transformed Switchboard gold parse trees with ELMo word representations (f-score=88.3%).
* [`swb_elmo_Edev.0.872.pt`](https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swb_elmo_Edev.0.872.pt): Our best model trained on the Switchboard gold parse trees with ELMo word representations (f-score=87.2%).

### Using the Trained Models
Use our trained models to find a constituency parse tree as well as disfluency labels for your own sentences. Befor running the following commands, make sure you meet the requirements for training and preparation. The format of the input in *raw_sentences.txt* is one sentence per line. For the best performance, remove punctuations and split clitics ("I 'm" instead of "I'm").

```
$ cd best_models
$ wget https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swb_fisher_bert_Edev.0.9078.pt
$ cd ..
$ python3 src/main.py parse --input-path best_models/raw_sentences.txt --output-path best_models/parsed_sentences.txt --model-path-base best_models/swb_fisher_bert_Edev.0.9078.pt >best_models/out.log
```

### Training Instructions
```
$ python3 src/train_parser.py --config results/best_nopunct_nopw_config.json --eval-path results/eval.txt >results/out_and_error.txt
```
### Citation
If you use this model, please cite the following papers:
```
@inproceedings{jamshid-lou-2019-neural,
    title = {Neural Constituency Parsing of Speech Transcripts},
    author = {Jamshid Lou, Paria and Wang, Yufei and Johnson, Mark},
    booktitle = {Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
    month = {June},
    year = {2019},
    address = {Minneapolis, Minnesota},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/N19-1282},
    doi = {10.18653/v1/N19-1282},
    pages = {2756--2765}
}
```

```
@inproceedings{jamshid-lou-2020-neural,
    title = {Improving Disfluency Detection by Self-Training a Self-Attentive Model},
    author = {Jamshid Lou, Paria and Johnson, Mark},
    booktitle = {arXiv preprint arXiv:2004.05323v1},
    year = {2020},
    url = {https://arxiv.org/pdf/2004.05323.pdf}
}
```

### Contact
Paria Jamshid Lou <paria.jamshid-lou@hdr.mq.edu.au>

 
### Credits
The code for self-attentive parser and part of the README file are based on https://github.com/nikitakit/self-attentive-parser.


