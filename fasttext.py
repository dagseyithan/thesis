import fastText
import platform

if platform.system() == 'Linux':
    model = fastText.load_model('/home/sdag/PycharmProjects/thesis/fasttext_german_embeddings/cc.de.300.bin')
else:
    model = fastText.load_model('C:\\Users\\seyit\\PycharmProjects\\thesis\\fasttext_german_embeddings\\cc.de.300.bin')

def get_fasttext_embedding(text):
    return model.get_word_vector(text)