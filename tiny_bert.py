from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


path_token = 'Tiny_Bert_token'
tokenizer = BertTokenizer.from_pretrained(
        'mrm8488/bert-tiny-finetuned-sms-spam-detection')
tokenizer.save_pretrained(path_token)


path_model = 'Tiny_Bert_model'
model = BertForSequenceClassification.from_pretrained(
        'mrm8488/bert-tiny-finetuned-sms-spam-detection', num_labels=2)
model.save_pretrained(path_model)