# A basic implementation for SRL-based question generation

Please note this is only a basic implementation.
## 1. Install necessary libraries and pre-trained models
```shell script
# install libraries
pip install allennlp
pip install allennlp-models
pip install ordered-set
pip install spacy
pip install nltk
# downoad pre-trained models

wget https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz
# for DP and NER I'd suggest using Spacy instead of AllenNLP
wget https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz
wget https://storage.googleapis.com/allennlp-public-models/ner-model-2018.12.18.tar.gz
```

## 2. Pre-process XSUM dataset
```shell script
# extract linguistic features (SRL, DP, NER) and saved to xsum_linguistic_features.json
python preprocess.py 
```

## 3. Generate questions

```shell script
python generating_questions_from_summary.py
```

### Citation:
```bibtex
@inproceedings{lyu2021improving,
  title={Improving Unsupervised Question Answering via Summarization-Informed Question Generation},
  author={Lyu, Chenyang and Shang, Lifeng and Graham, Yvette and Foster, Jennifer and Jiang, Xin and Liu, Qun},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={4134--4148},
  year={2021}
}
```
