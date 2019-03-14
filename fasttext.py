import fastText
import platform
import configurations

if platform.system() == 'Linux':
    if configurations.LANGUAGE == 'GERMAN':
        model = fastText.load_model(configurations.LINUX_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH)
    elif configurations.LANGUAGE == 'ENGLISH':
        model = fastText.load_model(configurations.LINUX_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH)
else:
    if configurations.LANGUAGE == 'GERMAN':
        model = fastText.load_model(configurations.WINDOWS_FASTTEXT_GERMAN_EMBEDDINGS_MODEL_PATH)
    elif configurations.LANGUAGE == 'ENGLISH':
        model = fastText.load_model(configurations.WINDOWS_FASTTEXT_ENGLISH_EMBEDDINGS_MODEL_PATH)

print('fastText model has been loaded...')


def __get_fasttext_word_embedding(text):
    return model.get_word_vector(text)