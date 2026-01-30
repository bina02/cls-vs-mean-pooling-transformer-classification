# cls-vs-mean-pooling-transformer-classification
# Comparative Analysis: <CLS> Token vs. Mean Pooling in Transformer-based Classification

The goal of this project is to perform text classification using a custom-built Transformer Encoder and to conduct a **comparative analysis** between two prominent pooling strategies: **<CLS> token** and **Mean Pooling**. By isolating these methods, we aim to identify their unique characteristics and performance trade-offs.


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
Initial results showed minimal performance differences between the Mean Pooling and <CLS> Token models, as both achieved comparable accuracy. However, clear signs of overfitting emerged, with validation loss diverging from training loss. To address this and better differentiate the two methods, I increased the dropout rate by 0.1 to strengthen regularization. While this mitigated the overfitting, the performance gap remained narrow. Consequently, I increased the model depth to 12 layers to observe how each strategy handles information under architectural stress.

## 5.2 dropout=0.3 Model Hyperparameters
### 5.2.1 dropout=0.3 Model performance
<img width="1490" height="590" alt="dropout_0 3" src="https://github.com/user-attachments/assets/ca1578c2-f1a6-4e69-862d-cffc10e7839d" />

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Accuracy** | 0.8906 | **0.8982** |
| **Precision** | 0.89 | **0.90** |
| **Recall** | 0.89 | **0.90** |
| **F1-Score** | 0.89 | **0.90** |

### 5.2.2 Analysis
To address the initial overfitting, I increased the dropout rate, which successfully mitigated the gap between training and validation performance to some extent. However, even with improved regularization, there was no significant performance disparity observed between the Mean Pooling and <CLS> Token models in the shallow configuration. This led to the conclusion that a more complex architectural environment was needed to identify the unique characteristics of each method. Consequently, I increased the model depth to 12 layers to stress-test both pooling strategies and observe how they handle information across a deeper network.
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
At 12 layers, the Mean Pooling model's performance collapsed to an accuracy of 0.25, equivalent to a random guess. This suggests that in deeper networks, the arithmetic mean dilutes the semantic signal with accumulated noise. Conversely, the `<CLS>` Token model maintained relatively stable performance, demonstrating higher robustness. This indicates that the `<CLS>` method, powered by self-attention, is more effective at selectively aggregating task-relevant features and filtering out the noise that propagates through deeper architectures.

# 6. Performance in test dataset

## 6.1 Base Model

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Accuracy** | 0.8859 | **0.8926** |
| **Precision** | 0.89 | **0.89** |
| **Recall** | 0.89 | **0.89** |
| **F1-Score** | 0.89 | **0.89** |
## 6.2 dropout=0.3 Model

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Accuracy** | 0.8987 | **0.9021** |
| **Precision** | 0.90 | **0.90** |
| **Recall** | 0.90 | **0.90** |
| **F1-Score** | 0.90 | **0.90** |
## 6.3 dropout=0.3, n_layer=12 Model

| Metric | Mean Pooling Model | CLS Token Model |
| :--- | :---: | :---: |
| **Val Accuracy** | 0.2538 | **0.8308** |
| **Val Precision** | 0.19 | **0.83** |
| **Val Recall** | 0.25 | **0.83** |
| **Val F1-Score** | 0.11 | **0.83** |

## 6.4 Review
In the final evaluation on the test data, the **`<CLS>` Token model** demonstrated slightly superior performance compared to the Mean Pooling model, though the margin remained narrow in shallow configurations. However, the most significant finding occurred when the model depth was increased to **12 layers**. In this deep architecture, the **Mean Pooling model's performance collapsed**, performing no better than a **random classifier**. This highlights a critical limitation of mean pooling in deep Transformers, where the accumulation of noise across layers overwhelms the averaged semantic signal. In contrast, the `<CLS>` token's ability to selectively aggregate information through self-attention proved essential for maintaining model robustness as depth increased.

# 7. Conclution
In summary, my experiment reveals that **Mean Pooling fails to work in deep models (such as 12 layers)**. This happens because, as the model gets deeper, "noise" or useless information builds up. When we average all the words together, this noise drowns out the actual meaning of the text, leading to a complete failure in performance. On the other hand, **the `<CLS>` token remains effective even in deep layers** because it uses attention to pick only the important information for classification. Therefore, **for deep Transformer models, using the `<CLS>` token is a much more reliable choice** than simple mean pooling to keep the model's performance stable.

