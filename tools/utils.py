import csv
import emoji
import string
from datetime import datetime

def write_dict(DICTIONARIES_PATH, frequency_dict, freq_dict_filename, probability_dict=None, probability_dict_filename=''):
	frequency_dict_csv = csv.writer(open(DICTIONARIES_PATH + freq_dict_filename + '.csv', 'w', encoding='UTF-16'))
	frequency_dict_csv.writerow(['word', 'frequency'])
	for key, val in frequency_dict.items():
		frequency_dict_csv.writerow([key, val])

	if probability_dict is not None:
		probability_dict_csv = csv.writer(open(DICTIONARIES_PATH + probability_dict_filename + '.csv', 'w', encoding='UTF-16'))
		probability_dict_csv.writerow(['word', 'probability'])
		for key, val in probability_dict.items():
			probability_dict_csv.writerow([key, val])

# Punctuation, Numbers and Emojis
def is_other(word):
	def isfloat(value):
		try:
			float(value)
			return True
		except ValueError:
			return False

	if word in emoji.UNICODE_EMOJI or word.isnumeric() or isfloat(word):
		return True

	if '\n' in word:
		return True
		
	for c in word:
		if c in string.punctuation and c is not "\'":
			return True
	return False

def printStatus(status):
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print(f"[{current_time}] {status}")