import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import pandas as pd
from pickle import load
from numpy.random import rand
from numpy.random import shuffle

################################ let us load 140K sentences from IIT dataset for training/testing  ##########################

eng=open('/home/rohan/Downloads/parallel/IITB.en-hi-en.txt') #iit dataset for english
hin=open('/home/rohan/Downloads/parallel/IITB.en-hi-hi.txt') #iit dataset for hindi
eng_text=eng.readlines()
eng.close()
hin_text=hin.readlines()
hin.close()


#make a new file named hin.txt for writing 140K sentence pairs of English and Hindi.
#the format looks like Eng_sentence + "\t" (tab or 4 spaces) + Hin_sentence + "\n" (new line character)
#take sentences of length less than or equal to 15 as we are making a small translation model

file=open('/path/to/your/file/named_as/hin.txt', 'w+') #if working in Google Colab, just keep path as /hin.txt

cnt=0 # to keep a count of 140K
for i in range(0, len(eng_text)):
    if cnt<140001:
        if len(eng_text[i].split(' '))<=15 and len(hin_text[i].split(' '))<=15: 
            drop=False
            for letter in hin_text[i]:
                if ord(letter)>=65 and ord(letter)<=122:
                    drop=True
                    break
            if not drop:
                file.write(eng_text[i].strip()+"\t"+hin_text[i].strip()+"\n")
                cnt+=1
file.close()

##################################  loading of 140K sentences is done, now we start cleaning  ###############################

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs

# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in range(0, len(pair)):
            if line%2==0:
            # normalize unicode characters
                pair[line] = normalize('NFD', pair[line]).encode('ascii', 'ignore')
                pair[line] = pair[line].decode('UTF-8')
            # tokenize on white space
                pair[line] = pair[line].split()
            # convert to lowercase
                pair[line] = [word.lower() for word in pair[line]]
            # remove punctuation from each token
                pair[line] = [word.translate(table) for word in pair[line]]
            # remove non-printable chars form each token
                pair[line] = [re_print.sub('', w) for w in pair[line]]
            # remove tokens with numbers in them
                pair[line] = [word for word in pair[line] if word.isalpha()]
            # store as string
                clean_pair.append(' '.join(pair[line]))
            else:
                pair[line]=re.sub("[२३०८१५७९४६]", "", pair[line])
                pair[line]=re.sub("'", '', pair[line])
                pair[line] = [word.translate(table) for word in pair[line]]
                clean_pair.append(''.join(pair[line]))
        cleaned.append(clean_pair)
    return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load dataset
filename = 'hin.txt'
doc = load_doc(filename)
# split into english-hindi pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'en-hi.pkl')
# spot check
for i in range(10):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))

############################ cleaning of 140K sentences is done, now we start loading pickle files ##########################

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('en-hi.pkl')

# reduce dataset size
n_sentences = 140000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:135000], dataset[135000:]
# save
save_clean_data(dataset, 'en-hi-both.pkl')
save_clean_data(train, 'en-hi-train.pkl')
save_clean_data(test, 'en-hi-test.pkl')

######### pickle files are saved as train, test and both. Now we are done with creating train and test dataset #############
