from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def data_preparation(sms):
    raw_text = sms.lower()
    with open('tfidfvectorizer.pk', 'rb') as file:
        tfidfvectorizer = pickle.load(file)
    tfidfvectorizer.transform([raw_text])

    vocab_size = len(tfidfvectorizer.get_feature_names_out())

    with open('tokenizer.pk', 'rb') as file:
        tokenizer = pickle.load(file)

    sms_sequences = tokenizer.texts_to_sequences([sms])

    # Ajoutez du padding pour avoir des séquences de même longueur
    max_sequence_length = 1000  # Remplacez cette valeur par la longueur souhaitée
    sms_padded = pad_sequences(sms_sequences, maxlen=max_sequence_length)
    return sms_padded.shape

    

def tf_classification(sms):
    tf_model = load_model('modeleV1.h5')


if __name__ == "__main__":

    sms = 'Hello, how are you?'

    print(data_preparation(sms = sms))
