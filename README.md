# Semi-Supervised Code-Switch Detection

The source code for research paper 'Much Gracias: Semi-supervised Code-switch Detection forSpanish-English: How far can we get?' by Dana-Maria Iliescu & Rasmus Grand & Sara Qirko & Rob van der Goot - IT-University of Copenhagen - April 2021

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

pip install -U spacy=2.3.2
```

## Usage
1. Create frequency/probability dictionaries and train the models by running the folowing:
	- ```python train_probability.py <lang1> <lang2> <'frequency' or 'probability'>```
	- ```python train_ngrams_character.py <lang1> <lang2>```
	- ```python train_ngrams_word.py <lang1> <lang2>```
	- ```python train_viterbi_v1.py <lang1> <lang2>```
2. Do classification by running
	- ```python code_switching_*.py <lang1> <lang2> <evaluation-dataset>```
	- ```python code_switching_*_ngrams.py <lang1> <lang2> <evaluation-dataset> <n> for word and character n-grams```

## Adding a new language pair
1. Add the two-letter code and name of the language ```langs.py``` (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes), for each new language
2. Add training monolingual data in ```datasets``` folder with the name ```monolingual-<lang>```, for each new language
3. Add test bilingual data in ```datasets/bilingual-annotated``` folder with the name  ```<lang1>-<lang2>```
4. Create an empty folder in ```results/predictions``` folder with the name ```<lang1>-<lang2>```
5. Train and test the models
