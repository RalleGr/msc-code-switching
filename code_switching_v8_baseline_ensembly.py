import json

PREDICTIONS_PATH = './results/predictions/'

predictionsFileNames = [
	'predictions_probabilities.txt',
	'predictions_5_grams.txt',
	'predictions_word_2_grams.txt',
	'predictions_viterbi_v1.txt',
	'predictions_viterbi_v2.txt',
	'predictions_LDA_v1.txt',
	# 'predictions_LDA_v2.txt',
	'predictions_SVM.txt',
	'predictions_LogisticRegression.txt',
]

predictions = dict()
for file in predictionsFileNames:
	with open(PREDICTIONS_PATH + file) as f:
		pred = json.load(f)
		# Save prediction model name without .txt extension as key, actual predictions as value
		predictions[file.split('.')[0]] = pred

print(predictions['predictions_5_grams']['test']) # prints lang1