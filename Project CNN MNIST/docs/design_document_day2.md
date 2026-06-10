# MNIST Classification System
## Product Design Document


- Version: 1.0.0
- Status: Active Development
- Owner: Data Engineering KNguyen
- Last Update: 2026-06-08

---

- Version: 1.1.0
- Status: Active Development
- Owner: Data Engineering KNguyen
- Last Update: 2026-06-10

---

Ở Version: 1.1.0 sau khi đã có folder data với mục đích viết báo cáo, chuyển dữ liệu từ .csv sang .npy để model có thể đọc trực tiếp và giảm bộ nhớ để lưu trữ. 

Ngày 10/06 ta sẽ đi sâu vào src và tests. Và trước đó ta phải lên kế hoạch cho một Model Architecture.

---

# Day 2 - Model Foundation Planning

## Objective

Sau khi hoàn thành Data Pipeline, toàn bộ dữ liệu đã được chuẩn hóa, kiểm định và đóng gói dưới định dạng `.npy` để phục vụ quá trình huấn luyện.

Mục tiêu của ngày làm việc thứ hai không phải huấn luyện mô hình ngay lập tức mà là xây dựng nền tảng kiến trúc cho giai đoạn Modeling.

Các thành phần được thiết kế trong ngày này phải đảm bảo:

* Dễ bảo trì.
* Dễ mở rộng.
* Dễ kiểm thử.
* Có khả năng tái sử dụng cho nhiều phiên bản mô hình trong tương lai.

---

# Current Project Status

## Completed

### Data Pipeline

* Data Ingestion
* Data Validation
* Data Profiling
* Data Preprocessing
* Dataset Splitting
* Dataset Release

### Dataset Assets

```text
data/

raw/
interim/
processed/
reports/
```

### Training Dataset

```text
X_train.npy
y_train.npy

X_valid.npy
y_valid.npy

X_test.npy
y_test.npy
```

Dataset đã sẵn sàng cho giai đoạn Modeling.

---

# Day 2 Deliverables

## Source Code Architecture

Thiết lập cấu trúc source code phục vụ huấn luyện và đánh giá mô hình.

```text
src/

data/
models/
training/
evaluation/
utils/
```

---

## Testing Architecture

Thiết lập hệ thống kiểm thử tự động.

```text
tests/

data/
models/
training/
```

Mục tiêu:

* Kiểm tra dữ liệu đầu vào.
* Kiểm tra shape tensor.
* Kiểm tra forward pass.
* Kiểm tra training pipeline.
* Giảm rủi ro regression trong các phiên bản tiếp theo.

Data Flow:
```
src/data
        ↓
tests/data

src/models
        ↓
tests/models

src/training
        ↓
tests/training
```

---

# Model Architecture Planning

## Problem Definition

Bài toán:

```text
Input:
28 x 28 Grayscale Image

Output:
Digit Class (0-9)
```

Loại bài toán:

```text
Multi-Class Classification
```

Số lớp:

```text
10 Classes
```

---

# Baseline Model

Phiên bản đầu tiên tập trung vào:

```text
CNN Baseline v1
```

Mục tiêu:

* Dễ giải thích.
* Dễ debug.
* Dễ benchmark.
* Làm mốc đánh giá cho các phiên bản sau.

---

# Planned Architecture

```text
Input Image
    ↓
Convolution Block
    ↓
Activation
    ↓
Pooling
    ↓
Feature Extraction
    ↓
Fully Connected Layer
    ↓
Class Prediction
```

---

# Success Criteria

Mô hình được xem là sẵn sàng chuyển sang giai đoạn huấn luyện khi:

* Dataset Loader hoạt động ổn định.
* Forward Pass thành công.
* Unit Tests đạt 100%.
* Shape Validation hoàn tất.
* Model Summary được xác nhận.
* Không phát sinh lỗi dữ liệu đầu vào.

---