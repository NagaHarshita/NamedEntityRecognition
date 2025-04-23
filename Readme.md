# Named Entity Recognition (NER)

This project implements and compares BiLSTM-based models for Named Entity Recognition (NER), trained on the CoNLL-2003 dataset. It covers preprocessing, model design, training strategies, and performance evaluation.

---

## 📚 Dataset
- **CoNLL-2003** dataset for NER.
- Split into `train`, `dev`, and `test`.
- Removed empty lines and converted words/tags to indexed sequences.

---

## 🧪 Task 1: BiLSTM-based Model

### 🔧 Architecture
```python
BLSTM(
  embedding: Embedding(23700, 100)
  lstm: LSTM(100, 256, dropout=0.33, bidirectional=True)
  linear: Linear(512 → 128)
  elu: ELU(alpha=1.0)
  classifier: Linear(128 → 10)
)
```

### ⚙️ Training Details
- Optimizer: SGD (lr = 0.8), with LR scheduler
- Epochs: 100
- Batch size: 32
- Used weighted cross-entropy to handle class imbalance
- Padded values ignored in loss with `ignore_index = -1`

### 📈 Results
- **Accuracy**: 93.65%
- **F1 Score**: 65.15  
- **Per-class F1 Scores**:
  - LOC: 79.93
  - MISC: 68.67
  - ORG: 58.87
  - PER: 52.81

---

## 🧪 Task 2: BiLSTM with GloVe Embeddings

### 🔧 Architecture
```python
BLSTM_Glove(
  embedding: Embedding(400000, 100)
  lstm: LSTM(100, 256, dropout=0.33, bidirectional=True)
  linear: Linear(512 → 128)
  elu: ELU(alpha=1.0)
  classifier: Linear(128 → 9)
)
```

### ➕ Additional Feature
- Added a case sensitivity flag (1 if the word is uppercase).

### 📈 Results
- **Accuracy**: 93.31%
- **F1 Score**: 63.54  
- **Per-class F1 Scores**:
  - LOC: 79.58
  - MISC: 66.42
  - ORG: 55.82
  - PER: 50.82

---

## 📌 Key Takeaways
- Class imbalance significantly impacts performance (especially on `PER`, `ORG`)
- GloVe embeddings help but need careful tuning
- Padding and batch-handling are crucial for consistent training

---

## 🛠️ Hyperparameters
```text
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 0.8
```
