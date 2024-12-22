# DistilBERT for sentiment prediction

## Overview

In this project, we fine-tuned the `DistilBERT` model to predict sentiments for IMDb reviews. The dataset contains `50,000` rows (`https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`), with the target variable indicating either a `positive` or `negative` review. We used the `AutoTokenizer` from Hugging Face to tokenize the text. The tokenizer works by converting raw text into a format that the model can understand, typically by breaking the text into smaller units (tokens) and mapping them to numerical representations using a pre-trained vocabulary. This tokenized data is then fed into the model.

We employed the `AutoModelForSequenceClassification` model, which is specifically designed for text classification tasks. This model includes a pre-trained DistilBERT architecture, which consists of a series of transformer layers. The model processes the tokenized input through its embedding and transformer layers, extracting contextual information from the text. Finally, it uses a classification head (a linear layer) to map the output to the desired classes: positive or negative sentiment. After fine-tuning the model on the dataset, we achieved a final accuracy of `93 %` on the test set.

## Model architecture

```
DistilBertForSequenceClassification(
  (distilbert): DistilBertModel(
    (embeddings): Embeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (layer): ModuleList(
        (0-5): 6 x TransformerBlock(
          (attention): DistilBertSdpaAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (q_lin): Linear(in_features=768, out_features=768, bias=True)
            (k_lin): Linear(in_features=768, out_features=768, bias=True)
            (v_lin): Linear(in_features=768, out_features=768, bias=True)
            (out_lin): Linear(in_features=768, out_features=768, bias=True)
          )
          (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (ffn): FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (activation): GELUActivation()
          )
          (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
  )
  (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

## Usage
Run the API server in its directory with `uvicorn app:app --host 0.0.0.0 --port 8000` , and test it by executing `./predict_sentiment.sh` to send a user review and receive predicted sentiment, to train the model or inference run `train.py` and `infer.py` respectively.


