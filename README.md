# cls-vs-mean-pooling-transformer-classification
이 프로젝트는 transformer모델에서 <b>cls와 mean_pooling 방식으로 분류</b> 작업을 하고, 두 개가 어떤 특징을 가지고 있는지, 비교 분석하는 프로젝트입니다.


# 1. Concept Overview
Classification using Transformers can be broadly divided into two main categories:

1. **Classification using the `<CLS>` token**
2. **Classification using `mean_pooling`**

## 1.1 The `<CLS>` Token Method

In the **`<CLS>` method**, a special `<CLS>` token is inserted at the beginning of a sentence. Through the **self-attention** mechanism, the `<CLS>` token captures the contextual information of the entire sequence. This token's final representation is then used to classify the sentence.
</p>

      cls_vector = x[:, 0, :]
      return self.linear(cls_vector)

## 1.2 Mean pooling
**Mean pooling** is a method where the sentence is classified using a vector derived by calculating the **arithmetic mean of the row vectors** (token embeddings) from the final output of the encoder.

      x = x * mask.unsqueeze(-1).float()
      x = x.sum(dim=1)
      count_x = mask.sum(dim=-1).unsqueeze(-1).float()
      x = x / count_x
      return self.linear(x)

# 2. Dataset: AGNews
**AGNews (AG's News Corpus)** is a widely used benchmark dataset for text classification tasks. It consists of news articles gathered from more than 2,000 news sources.

## 2.1 Overview
The dataset is specifically designed for **topic classification**, where models must categorize news articles into one of four distinct classes. Each sample contains a **Title** and a **Description** of the article.

### 2.1.1 Dataset Statistics
* **Total Classes**: 4
* **Training Samples**: 120,000 (30,000 per class)
* **Testing Samples**: 7,600 (1,900 per class)
* **Data Format**: CSV (Class Index, Title, Description)

### 2.1.2 Class Labels
The articles are categorized into the following four topics:
1.  **World** (Index 1)
2.  **Sports** (Index 2)
3.  **Business** (Index 3)
4.  **Sci/Tech** (Index 4)

### 2.1.3 Task Goal
The goal is to train a model to accurately predict the class index ($y \in \{1, 2, 3, 4\}$) based on the input text ($x$), which is typically the concatenation of the title and description.

## 2.2 Data Preprocessing

### 2.2.1 Creation of 'full text' and 'Class' Columns
The original CSV file consists of `Class Index`, `Title`, and `Description`.
* **Class Column:** Since the `Class Index` values range from 1 to 4, we subtract 1 from each value to map them to a **0 to 3** range for model compatibility.
* **Full Text:** For the classification task, the `Title` and `Description` are concatenated into a single field named **full text**.

### 2.2.2 Conversion to .txt Format
In this project, we use **SentencePiece** as our tokenizer.
Since SentencePiece requires raw text files for training or processing, the CSV data is converted into a **.txt** format.

### 2.2.3 Training, Validation, and Test Sets
The dataset is divided into three parts:
1. **Test Set:** Pre-separated in the original dataset.
2. **Training & Validation Sets:** The original training data was split into training and validation sets with a **validation ratio of 0.2** (80/20 split).

# 3. Tokenizer
## 3.1 Why I choose Sentencepiece as Tokenizer
I chose **SentencePiece** as the tokenizer for this project for the following reasons:

* **Practical Familiarity:** It is the tokenizer I am most proficient with, allowing for stable and efficient implementation.
* **Handling Out-of-Vocabulary (OOV) Words:** As a **subword tokenizer**, SentencePiece can effectively decompose and represent unseen words or neologisms (newly coined words).
* **Suitability for News Data:** News articles frequently contain new terminology, slang, or proper nouns. SentencePiece is well-suited for this environment as it breaks these down into meaningful subword units rather than treating them as unknown tokens.
## 3.2 Tokenizer Hyperparameters
The SentencePiece tokenizer was trained using the following configuration:

      spm.SentencePieceTrainer.Train(
    input='/content/drive/MyDrive/github/cls-vs-mean-pooling-transformer-classification/data/processed/train.csv',
    model_prefix = '/content/drive/MyDrive/github/cls-vs-mean-pooling-transformer-classification/data/processed/tokenizer',
    vocab_size = 8000,
    model_type='unigram',
    user_defined_symbols=['[CLS]', '[SEP]'],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )

The vocabulary size was strategically set to **8,000** based on the following technical considerations:

* **Prevention of Overfitting**: An excessively large vocabulary significantly increases the number of parameters in the embedding matrix. This can lead to overfitting, where the model memorizes rare tokens rather than learning generalizable patterns, especially given the specific scale of our dataset.
* **Computational Efficiency**: Keeping the vocabulary size compact reduces the overall memory footprint and lowers the computational overhead during both training and inference, leading to faster iteration cycles.
* **Optimal Semantic Balance**: A vocab size of 8,000 provides a sufficient linguistic bottleneck to represent the core themes of the AGNews dataset effectively while maintaining model agility and robustness.

# 4. Model
![alt text](image.png)
The model was implemented from scratch, following the Encoder structure of the Transformer. The layers consist of:

1. **Positional Encoding**
2. **Multi-head Attention**
3. **Add & Norm**
4. **Feed Forward**
5. **Add & Norm**

Additionally, the `forward` function was designed to toggle between **Mean Pooling** and the **<CLS> token** method using a boolean flag (`is_mean_pooling`). This allows for a direct comparison between the two strategies within the same architecture.

        def forward(self, x, mask, is_mean_pooling=True):
          x = self.embedding(x)
          x = self.positionalEncoding(x)
          for layer in self.layers:
            x = layer(x, mask)
      
          if is_mean_pooling:
            x = x * mask.unsqueeze(-1).float()
            x = x.sum(dim=1)
            count_x = mask.sum(dim=-1).unsqueeze(-1).float()
            x = x / count_x
            return self.linear(x)
      
          else:
            cls_vector = x[:, 0, :]
            return self.linear(cls_vector)

All models used in this project follow this exact architecture, with the only differences being the hyperparameters.
## 4.1 Base Model Hyperparameters
    "vocab_size": 8000,
    "d_model": 256,
    "max_len": 256,
    "n_head": 8,
    "d_ff": 1024,
    "n_layers": 8,
    "num_class": 4,
    "dropout": 0.2

    "train_batch_size": 8,
    "val_batch_size": 8,
    "test_batch_size": 4
    
## 4.2 dropout=0.3 Model Hyperparameters
    "vocab_size": 8000,
    "max_len": 256,
    "n_head": 8,
    "d_ff": 1024,
    "n_layers": 8,
    "num_class": 4,
    "dropout": 0.3

    "train_batch_size": 8,
    "val_batch_size": 8,
    "test_batch_size": 4

## 4.3 dropout=0.3, n_layer=12 Model Hyperparameters
    "vocab_size": 8000,
    "max_len": 256,
    "n_head": 8,
    "d_ff": 1024,
    "n_layers": 12,
    "num_class": 4,
    "dropout": 0.3

    "train_batch_size": 8,
    "val_batch_size": 8,
    "test_batch_size": 4
    
# 5. Model performance & Analysis
## 5.1 Base Model
### 5.1.1 Base Model performance
<img width="1490" height="590" alt="base_model" src="https://github.com/user-attachments/assets/558d9d2d-5ded-4a79-a09b-6a0f41b8d810" />

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Val Accuracy** | **0.9031** | 0.9018 |
| **Val Precision** | **0.90** | 0.90 |
| **Val Recall** | **0.90** | 0.90 |
| **Val F1-Score** | **0.90** | 0.90 |

### 5.1.2 Analysis
전체적으로 봤을 때, 이 경우, Mean Pooling Model과 CLS Token Model과의 성능차이가 거의 없었다.
하지만 그래프를 봤을 때, 과대적합의 양상이 보였으므로, dropout을 0.1 늘리고 다시 비교해볼 계획이다.

## 5.2 dropout=0.3 Model Hyperparameters
### 5.2.1 dropout=0.3 Model performance
<img width="1490" height="590" alt="dropout_0 3" src="https://github.com/user-attachments/assets/ca1578c2-f1a6-4e69-862d-cffc10e7839d" />

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Val Accuracy** | 0.2552 | **0.8302** |
| **Val Precision** | 0.20 | **0.83** |
| **Val Recall** | 0.26 | **0.83** |
| **Val F1-Score** | 0.11 | **0.83** |

### 5.2.2 Analysis

## 5.3 dropout=0.3, n_layer=12 Model Hyperparameters
## 5.3.1 dropout=0.3, n_layer=12 Model performance
<img width="1489" height="590" alt="dropout_0 3_n_layer=12" src="https://github.com/user-attachments/assets/954bcdf5-0052-4b60-8961-836feb8bdf42" />

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Val Accuracy** | 0.2552 | **0.8302** |
| **Val Precision** | 0.20 | **0.83** |
| **Val Recall** | 0.26 | **0.83** |
| **Val F1-Score** | 0.11 | **0.83** |

### 5.3.2 Analysis

