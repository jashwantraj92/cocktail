model = TFBertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
print(output)"""
(myenv) root@gpu:/home/cc/cocktail# cat text-classification/sentiment.py
from transformers import pipeline
import sys
classifier = pipeline('sentiment-analysis',model=sys.argv[1])
sentences=['We are very happy to show you the 🤗 Transformers library.','thi is not good','im am really happy', ' he is a bad guy']
for sentence in sentences:
    print(classifier(sentence))
