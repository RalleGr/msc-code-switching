# Semi-Supervised Code-Switch Detection

The source code for research paper ***

## Requirements
- Common Python 3 libraries
```sh
pip install emoji regex bs4 sklearn pandas numpy matplotlib 
```
- nltk library
```sh
pip install --user -U nltk
```

- SpaCy library
```sh
pip install -U pip setuptools wheel

pip install -U spacy
```

## Usage
1. Add language two-letter code (```<lang>```) and name in the ```langs.py```
2. Add training monolingual data in ```datasets``` folder with the name ```monolingual-<lang>```
3. Create frequency/probability dictionaries and train the models by running the folowing:
	- ```python train_probability.py <lang1> <lang2>```
	- ```python train_ngrams_character.py <lang1> <lang2>```
	- ```python train_ngrams_word.py <lang1> <lang2>```
	- ```python train_viterbi_v1.py <lang1> <lang2>```
4. Do classification by running
	- ```python code_switching_*.py <evaluation-dataset>```
	- ```python code_switching_*.py <n> <evaluation-dataset> for word and character n-grams```
