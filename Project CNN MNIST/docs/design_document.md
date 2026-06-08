# MNIST Classification System
## Product Design Document


- Version: 1.0.0
- Status: Active Development
- Owner: Data Engineering KNguyen
- Last Update: 2026-06-08

---


## 1. Executive Summary

MNIST Classification System là một hệ thống Machine Learning được xây dựng nhằm tự động nhận diện chữ số viết tay từ ảnh đầu vào.

Hệ thống nhận ảnh grayscale kích thước 28×28 và dự đoán một trong mười lớp:

```text
0 1 2 3 4 5 6 7 8 9
```

Dự án được thiết kế như một sản phẩm Machine Learning hoàn chỉnh thay vì một notebook nghiên cứu đơn lẻ.

Mục tiêu của hệ thống là cung cấp:

* Dataset Management
* Training Pipeline
* Evaluation Pipeline
* Inference Pipeline
* Experiment Tracking
* Model Versioning
* Reproducible Training

---

## 2. Problem Statement

Việc nhận dạng chữ số viết tay là một bài toán phân loại ảnh cơ bản nhưng đại diện cho hầu hết các thành phần quan trọng của một hệ thống Computer Vision.

MNIST được sử dụng làm môi trường kiểm chứng cho:

* Data Pipeline
* Model Training
* Model Evaluation
* Inference Workflow
* Experiment Management

Hệ thống này đóng vai trò là nền tảng trước khi mở rộng sang các tập dữ liệu lớn hơn như:

* Fashion MNIST
* CIFAR-10
* SVHN
* EMNIST

---

## 3. Objectives

### Primary Objectives

Xây dựng một pipeline có khả năng:

* Train model từ dữ liệu thô
* Đánh giá hiệu năng
* Lưu checkpoint
* Tái tạo kết quả
* Triển khai suy luận

---

### Secondary Objectives

Cho phép:

* So sánh nhiều phiên bản mô hình
* Theo dõi lịch sử huấn luyện
* Quản lý thực nghiệm
* Mở rộng sang dataset khác

---

## 4. Scope

### Included

#### Dataset

* MNIST Train Set
* MNIST Test Set

#### Training

* Data Loading
* Data Normalization
* Model Training
* Checkpoint Saving

#### Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

#### Inference

* Single Image Prediction
* Batch Prediction

#### Tracking

* Metrics Logging
* Artifact Storage
* Model Versioning

---

### Excluded

* Distributed Training
* Multi-GPU Training
* Hyperparameter Search
* Online Serving
* Production Deployment

---

## 5. Success Criteria

Một phiên bản được xem là thành công khi:

| Metric                | Target |
| --------------------- | ------ |
| Accuracy              | > 98%  |
| Reproducibility       | 100%   |
| Training Failure Rate | < 1%   |
| Model Loading Success | 100%   |

---

## 6. System Architecture

```text
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Training Pipeline
    │
    ▼
Model Checkpoint
    │
    ├────────► Evaluation Pipeline
    │
    └────────► Inference Pipeline
```

---

## 7. Data Flow

### Input

```text
28 x 28 grayscale image
```

---

### Preprocessing

```text
Load Image
    ↓
Normalize
    ↓
Tensor Conversion
```

---

### Output

```text
Predicted Digit
```

Ví dụ:

```json
{
  "prediction": 7,
  "confidence": 0.992
}
```

---

## 8. Repository Structure

```text
mnist-classifier/

├── configs/
├── data/
├── artifacts/
├── notebooks/
├── src/
├── tests/
│
├── train.py
├── evaluate.py
├── predict.py
│
├── requirements.txt
└── README.md
```

---

## 9. Artifact Management

Mỗi lần huấn luyện sinh ra một experiment độc lập.

```text
artifacts/

└── run_001/
    ├── config.yaml
    ├── metrics.json
    ├── checkpoint.npz
    ├── loss_curve.png
    └── confusion_matrix.png
```

Mục tiêu:

* Reproducibility
* Auditability
* Experiment Comparison

---

## 10. Configuration Strategy

Tất cả thông số huấn luyện được quản lý tập trung.

Ví dụ:

```yaml
batch_size: 64
epochs: 20
learning_rate: 0.001
optimizer: adam
```

Không hard-code tham số bên trong source code.

---

## 11. Model Versioning

### v1.0.0

Initial Release

Features:

* CNN Classifier
* Training Pipeline
* Evaluation Pipeline
* Inference Pipeline

---

### v1.1.0

Planned

Features:

* Better Logging
* Early Stopping
* Learning Rate Scheduler

---

### v1.2.0

Planned

Features:

* Fashion MNIST Support

---

### v2.0.0

Planned

Features:

* CIFAR-10 Support
* Multi-class Image Classification

---

## 12. Experiment Lifecycle

```text
Create Config
      │
      ▼
Train Model
      │
      ▼
Save Checkpoint
      │
      ▼
Evaluate
      │
      ▼
Generate Reports
      │
      ▼
Register Version
```

---

## 13. Risk Assessment

| Risk                    | Impact |
| ----------------------- | ------ |
| Overfitting             | High   |
| Data Corruption         | Medium |
| Checkpoint Loss         | Medium |
| Reproducibility Failure | High   |

---

## 14. Future Roadmap

### Phase 1

MNIST Classification

Status: Active

---

### Phase 2

Fashion MNIST

Status: Planned

---

### Phase 3

CIFAR-10

Status: Planned

---

### Phase 4

Generic Image Classification Framework

Status: Vision

---

## 15. Deliverables

Phiên bản hoàn thiện phải cung cấp:

* Reproducible Training Pipeline
* Evaluation Reports
* Saved Checkpoints
* Prediction Interface
* Experiment History
* Versioned Models
* Technical Documentation

Mọi kết quả huấn luyện phải có khả năng tái tạo từ cấu hình và dữ liệu đầu vào tương ứng.
