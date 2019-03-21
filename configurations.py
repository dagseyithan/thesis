import configparser

config = configparser.ConfigParser()
config.read('config.ini')

LANGUAGE = config['GENERAL']['language']

LINUX_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['fasttext_german_embeddings_model_path']
LINUX_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['fasttext_english_embeddings_model_path']
WINDOWS_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['fasttext_german_embeddings_model_path']
WINDOWS_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['fasttext_english_embeddings_model_path']
LINUX_ELMO_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['elmo_german_embeddings_model_path']
LINUX_ELMO_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['elmo_english_embeddings_model_path']
WINDOWS_ELMO_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['elmo_german_embeddings_model_path']
WINDOWS_ELMO_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['elmo_english_embeddings_model_path']
WINDOWS_TIGER_CORPUS_PATH = config['PATH_WINDOWS']['tiger_corpus_path']