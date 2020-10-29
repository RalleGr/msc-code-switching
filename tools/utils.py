import csv

def write_dict(DICTIONARIES_PATH, frequency_dict, lang, probability_dict=None):
	frequency_dict_csv = csv.writer(open(DICTIONARIES_PATH+'frequency_dict_' + lang + '.csv', 'w',encoding='UTF-16'))
	frequency_dict_csv.writerow(['word', 'frequency'])
	for key, val in frequency_dict.items():
		frequency_dict_csv.writerow([key, val])

	if probability_dict is not None:
		probability_dict_csv = csv.writer(open(DICTIONARIES_PATH +'probability_dict_' + lang + '.csv', 'w',encoding='UTF-16'))
		probability_dict_csv.writerow(['word', 'probability'])
		for key, val in probability_dict.items():
			probability_dict_csv.writerow([key, val])