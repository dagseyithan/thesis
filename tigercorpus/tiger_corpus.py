from config import configurations
import csv
import nltk


doc_dict = {}

with open('documents.tsv') as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  for row in reader:
      doc_dict[str(int(row[1]) - 1)] = row[0]

print(doc_dict['0'])
print(doc_dict['1'])
print(doc_dict['2'])

corp = nltk.corpus.ConllCorpusReader('.', configurations.TIGER_CORPUS_FILE, ['ignore', 'words', 'ignore', 'ignore', 'pos'], encoding='utf-8')
words = list(corp.words())
sents = list(corp.sents())