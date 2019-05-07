import configparser
import os

config = configparser.ConfigParser()

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
config.read(config_path)

LANGUAGE = config['GENERAL']['language']
FASTTEXT_VECTOR_LENGTH = int(config['GENERAL']['fasttext_vector_length'])
ELMO_VECTOR_LENGTH = int(config['GENERAL']['elmo_vector_length'])
MAX_TEXT_WORD_LENGTH = int(config['GENERAL']['max_text_word_length'])
TIGER_CORPUS_FILE = config['GENERAL']['tiger_corpus_file']

LINUX_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['fasttext_german_embeddings_model_path']
LINUX_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['fasttext_english_embeddings_model_path']
WINDOWS_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['fasttext_german_embeddings_model_path']
WINDOWS_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['fasttext_english_embeddings_model_path']
LINUX_ELMO_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['elmo_german_embeddings_model_path']
LINUX_ELMO_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_LINUX']['elmo_english_embeddings_model_path']
WINDOWS_ELMO_GERMAN_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['elmo_german_embeddings_model_path']
WINDOWS_ELMO_ENGLISH_EMBEDDINGS_MODEL_PATH = config['PATH_WINDOWS']['elmo_english_embeddings_model_path']
WINDOWS_TIGER_CORPUS_PATH = config['PATH_WINDOWS']['tiger_corpus_path']
LINUX_TIGER_CORPUS_PATH = config['PATH_LINUX']['tiger_corpus_path']
WINDOWS_ORIGINAL_PRODUCTS_FILE_PATH = config['PATH_WINDOWS']['original_products_file_path']
LINUX_ORIGINAL_PRODUCTS_FILE_PATH = config['PATH_LINUX']['original_products_file_path']
