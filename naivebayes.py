import os
import json
import re
from collections import defaultdict

def getExpanded(token):
	contractions = {
		"i'm": "i am",
		"can't": "cannot",
		"could've": "could have",
		"couldn't": "could not",
		"didn't": "did not",
		"doesn't": "does not",
		"don't": "do not",
		"hadn't": "had not",
		"hasn't": "has not",
		"haven't": "have not",
		"how'd": "how did",
		"he'll": "he will",
		"aren't": "are not",
		"how'll": "how will",
		"i'll": "i will",
		"i've": "i have",
		"isn't": "is not",
		"it'll": "it will",
		"it's": "it is",
		"let's": "let us",
		"might've": "might have",
		"must've": "must have",
		"she'll": "she will",
		"she's": "she is",
		"should've": "should have",
		"shouldn't": "should not",
		"that's": "that is",
		"there's": "there is",
		"they're": "they are",
		"they've": "they have",
		"we'll": "we will",
		"we're": "we are",
		"we've": "we have",
		"weren't": "were not",
		"what'll": "what will",
		"what're": "what are",
		"what's": "what is",
		"what've": "what have",
		"when's": "when is",
		"where'd": "where did",
		"where's": "where is",
		"who'll": "who will",
		"who's": "who is",
		"who've": "who have",
		"why's": "why is",
		"won't": "will not",
		"would've": "would have",
		"wouldn't": "would not",
		"y'all": "you all",
		"you'll": "you will",
		"you're": "you are",
		"you've": "you have"
	}
	return contractions.get(token)

def tokenizeText(text):
	tokens = text.split()
	new_tokens = []
	for token in tokens:
		if '\'' in token:
			expanded = getExpanded(token)
			if expanded:
				new_tokens.extend(expanded.split())
				continue
		if '-' in token:
			if token.startswith('-'):
				token = token[1:]
			if token.endswith('-'):
				token = token[:-1]
		new_tokens.append(token)
	return new_tokens

def trainNaiveBayes(trainingfiles, tokenized_text):
	vocab = set()
	class_counts = defaultdict(int)
	word_counts = defaultdict(lambda: defaultdict(int))

	for file in trainingfiles:
		label = re.sub(r'\d+\.txt$', '', file.split('/')[-1])
		class_counts[label] += 1
		file_name = os.path.basename(file).split('.')[0]
		for word in tokenized_text[file_name]:
			vocab.add(word)
			word_counts[label][word] += 1

	total_docs = sum(class_counts.values())
	class_probs = {label: count/total_docs for label, count in class_counts.items()}

	vocab_size = len(vocab)
	word_probs = defaultdict(lambda: defaultdict(float))
	for label in class_counts.keys():
		num_words = sum(word_counts[label].values())
		for word in vocab:
			word_probs[label][word] = (word_counts[label][word]+1)/(num_words+vocab_size)

	return class_probs, word_probs, vocab_size
   
def testNaiveBayes(test_file, class_probs, word_probs, vocab_size, tokenized_text):
	file_name = os.path.basename(test_file).split('.')[0]
	likelihoods = defaultdict(float)
	for label, prob in class_probs.items():
		likelihoods[label] = prob
		for word in tokenized_text[file_name]:
			if word in word_probs[label]:
				likelihoods[label] *= word_probs[label][word]
			else:
				likelihoods[label] *= 1/(sum(word_probs[label].values())+vocab_size)
	predicted_class = max(likelihoods, key=likelihoods.get)
	return predicted_class

def main():
	# Create the training files from trainfile-2.json
	if not os.path.exists("songs_train"):
		os.makedirs("songs_train")
		with open("trainfile-2.json", 'r') as f:
			content = f.read()
		data = json.loads(content)
		for i, entry in enumerate(data):
			lyric, artist = entry
			file_path = f"songs_train/{artist}{i}.txt"
			with open(file_path, 'w') as f:
				try: f.write(lyric)
				except: pass

	# Create the testing files from testfile-2.json
	if not os.path.exists("songs_test"):
		os.makedirs("songs_test")
		with open("testfile-2.json", 'r') as f:
			content = f.read()
		data = json.loads(content)
		for i, entry in enumerate(data):
			lyric, artist = entry
			file_path = f"songs_test/{artist}{i}.txt"
			with open(file_path, 'w') as f:
				try: f.write(lyric)
				except: pass

	train_files = [os.path.join("songs_train/", file) for file in os.listdir("songs_train/")]
	test_files = [os.path.join("songs_test/", file) for file in os.listdir("songs_test/")]

	tokenized_train_text = defaultdict(list)
	tokenized_test_text = defaultdict(list)

	for file in train_files:
		with open(file, 'r') as f:
			file_name = os.path.basename(file).split('.')[0]
			content = f.read()
			tokenized_train_text[file_name] = tokenizeText(content)

	for file in test_files:
		with open(file, 'r') as f:
			file_name = os.path.basename(file).split('.')[0]
			content = f.read()
			tokenized_test_text[file_name] = tokenizeText(content)

	class_probs, word_probs, vocab_size = trainNaiveBayes(train_files, tokenized_train_text)

	with open("naivebayes.output", 'w') as out_f:
		out_f.write("[Real Artist] | [Predicted Artist] | [Accuracy] | [Lyric]\n\n")

		correct_predictions = 0
		for file_path in test_files:
			filename = os.path.basename(file_path)
			predicted_class = testNaiveBayes(file_path, class_probs, word_probs, vocab_size, tokenized_test_text)
			true_class = re.sub(r'\d+\.txt$', '', filename)
			lyric = ' '.join(tokenized_test_text[filename.split('.')[0]])
			out_f.write(f"{true_class} | { predicted_class} | {predicted_class==true_class} | {lyric} \n")
			if predicted_class == true_class:
				correct_predictions += 1

		accuracy = correct_predictions / len(test_files)
		out_f.write(f"\nAccuracy: {accuracy}\n")
		out_f.write(f"Correct Predictions: {correct_predictions}\n")
		out_f.write(f"Total Predictions: {len(test_files)}\n")

if __name__ == "__main__":
	main()