import configparser
import os

config = configparser.ConfigParser()

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
config.read(config_path)

LANGUAGE = config['GENERAL']['language']
FASTTEXT_VECTOR_LENGTH = int(config['GENERAL']['fasttext_vector_length'])
ELMO_VECTOR_LENGTH = int(config['GENERAL']['elmo_vector_length'])
MAX_TEXT_WORD_LENGTH = int(config['GENERAL']['max_text_word_length'])
MAX_WORD_CHARACTER_LENGTH = int(config['GENERAL']['max_word_character_length'])
ALPHABET_LENGTH = int(config['GENERAL']['alphabet_length'])
EPSILON = float(config['GENERAL']['epsilon'])
TIGER_CORPUS_FILE = config['GENERAL']['tiger_corpus_file']
EMBEDDER = config['GENERAL']['embedder']
BATCH_SIZE = int(config['GENERAL']['batch_size'])

if EMBEDDER == 'FASTTEXT':
    EMBEDDING_LENGTH = FASTTEXT_VECTOR_LENGTH
else:
    EMBEDDING_LENGTH = ELMO_VECTOR_LENGTH

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
WINDOWS_DATASET_FILE_PATH = config['PATH_WINDOWS']['dataset_file_path']
LINUX_DATASET_FILE_PATH = config['PATH_LINUX']['dataset_file_path']