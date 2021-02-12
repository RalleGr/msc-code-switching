# Semi-Supervised Code-Switch Detection

The source code for research paper ***

## Requirements
- Common Python 3 libraries
```sh
pip install emoji beautifulsoup4 sklearn pandas numpy matplotlib 
```
- SpaCy library
```sh
pip install -U pip setuptools wheel

pip install -U spacy
```


## How to run
1. Create the dicionaries and train the models by running the folowing:
	- ```python train_probability.py ```
	- ```python train_ngrams_character.py ```
	- ```python train_ngrams_word.py ```
	- ```python train_viterbi_v1.py ```
2. Do classification by running
	- ```python code_switching_*.py <n> <evaluation-dataset>```

	Where `<n>` is the n-gram parameter
