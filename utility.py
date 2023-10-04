from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


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
    return sms_padded

    

def tf_classification(sms_padded):
    tf_model = load_model('modeleV1.h5')
    tf_pred = tf_model.predict(sms_padded)
    if tf_pred <= 0.5:
        tf_pred = 'ham'
    else:
        tf_pred = 'spam'

    return tf_pred


def mlp_preparation(sms):
    with open('tfidfvecto_mlp.pkl', 'rb') as file:
        tfidfvectorizer = pickle.load(file)
    sms_tfidf = tfidfvectorizer.transform([sms])

    return sms_tfidf


def mlp_classification(sms_tfidf):
    with open('mlp.pkl', 'rb') as file:
        mlp_model = pickle.load(file)
    return mlp_model.predict(sms_tfidf)[0]


def nb_clasification(sms_tfidf):
    with open ('nb.pkl', 'rb') as file:
        nb_model = pickle.load(file)
    return nb_model.predict(sms_tfidf.toarray())[0]


def bert_pretrained(sms):
    tokenizer = BertTokenizer.from_pretrained(
        'Tiny_Bert_token')
    model = BertForSequenceClassification.from_pretrained(
        'Tiny_Bert_model', num_labels=2)
    sms_tokens = tokenizer(
        [sms], padding=True, truncation=True, return_tensors='pt')
    

    model.eval()
    with torch.no_grad():
        outputs = model(**sms_tokens)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

        if 1 in predicted_labels:
            return 'spam'
        elif 0 in predicted_labels:
            return 'ham'



if __name__ == "__main__":

    sms = 'reply to win £100 weekly! where will the 2006 fifa world cup be held? send stop to 87239 to end service'
    sms_tfidf = mlp_preparation(sms=sms)


    print(bert_pretrained(sms=sms))
