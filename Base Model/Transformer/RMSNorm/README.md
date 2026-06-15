# RMSNorm (Root Mean Square Layer Normalization)

<img src="assets/intro.png">

## 1. Giới thiệu

Trong các mạng học sâu hiện đại, đặc biệt là Transformer và Large Language Models (LLMs), việc kiểm soát độ lớn của activation là một vấn đề quan trọng.

Khi số lượng layer tăng lên hàng chục hoặc hàng trăm tầng, activation có xu hướng:

* Tăng quá lớn (Exploding Activations)
* Suy giảm quá nhỏ (Vanishing Activations)
* Gây mất ổn định trong quá trình lan truyền gradient

Để giải quyết vấn đề này, các phương pháp Normalization được đưa vào kiến trúc mạng nhằm duy trì sự ổn định của tín hiệu trong quá trình huấn luyện.

Trong Transformer gốc, phương pháp được sử dụng là Layer Normalization (LayerNorm). Tuy nhiên, các nghiên cứu sau đó chỉ ra rằng việc chuẩn hóa theo trung bình (mean-centering) có thể không phải thành phần thiết yếu để đạt được tính ổn định.

Từ nhận xét này, RMSNorm được đề xuất như một phiên bản đơn giản hơn của LayerNorm nhưng vẫn giữ được các đặc tính quan trọng đối với việc huấn luyện mô hình quy mô lớn.

---

## 2. Động cơ nghiên cứu

Cho vector đầu vào:

$$
x = [x_1, x_2, \ldots, x_d]
$$

LayerNorm thực hiện hai bước:

### Bước 1: Loại bỏ giá trị trung bình

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
x \rightarrow x - \mu
$$

### Bước 2: Chuẩn hóa phương sai

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

$$
x \rightarrow \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Như vậy LayerNorm vừa:

* Dịch chuyển vector về tâm.
* Kiểm soát độ lớn vector.

Tuy nhiên các nghiên cứu thực nghiệm cho thấy:

> Việc kiểm soát độ lớn của vector quan trọng hơn việc đưa trung bình về 0.

Điều này dẫn tới ý tưởng: Thay vì chuẩn hóa theo mean và variance, chỉ cần chuẩn hóa theo độ lớn tổng thể của vector.

---

## 3. Ý tưởng cốt lõi của RMSNorm

RMSNorm loại bỏ hoàn toàn bước tính mean. Thay vào đó chỉ đo độ lớn của vector bằng Root Mean Square (Căn bậc hai của trung bình bình phương):

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}
$$

Sau đó chuẩn hóa trực tiếp:

$$
\hat{x} = \frac{x}{\text{RMS}(x)}
$$

Cuối cùng áp dụng hệ số học được (Scale):

$$
y = \gamma \odot \hat{x}
$$

Trong đó:
* $\gamma$ là vector tham số học được.
* $\odot$ là phép nhân từng phần tử (element-wise product).

Toàn bộ cơ chế của RMSNorm chỉ xoay quanh việc kiểm soát độ lớn của activation.

---

## 4. Công thức toán học đầy đủ

Cho $x \in \mathbb{R}^{d}$, tính RMS kèm số epsilon ($\epsilon$) để tránh lỗi chia cho 0:

$$
r = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

Chuẩn hóa vector:

$$
\hat{x} = \frac{x}{r}
$$

Scale với tham số trọng số:

$$
y = \gamma \odot \hat{x}
$$

Suy ra công thức tổng quát:

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

Khác với LayerNorm:
* Không tính mean ($\mu$).
* Không tính variance ($\sigma^2$).
* Không sử dụng tham số bias ($\beta$).

---

## 5. Diễn giải hình học

Chuẩn Euclidean ($L_2$ norm) của một vector được định nghĩa:

$$
\|x\|_2 = \sqrt{\sum_{i=1}^{d}x_i^2}
$$

Ta có mối liên hệ:

$$
\text{RMS}(x) = \frac{\|x\|_2}{\sqrt{d}}
$$

Do đó RMSNorm có thể viết lại dưới dạng tỷ lệ:

$$
\text{RMSNorm}(x) \propto \frac{x}{\|x\|_2}
$$

Điều này có nghĩa RMSNorm gần tương đương với việc đưa vector lên một hypersphere (siêu cầu) có bán kính cố định.

Các đặc tính quan trọng:
* Giữ nguyên hướng vector.
* Điều chỉnh độ lớn vector.
* Không thay đổi thông tin tương đối giữa các chiều biểu diễn.

Trong không gian biểu diễn, RMSNorm chủ yếu kiểm soát độ dài vector thay vì xoay hoặc dịch chuyển hướng biểu diễn của nó.

---

## 6. Tính chất Scale Invariance

Giả sử ta scale vector đầu vào với một hệ số $c > 0$ thành $x' = cx$. Khi đó:

$$
\text{RMS}(x') = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(cx_i)^2} = c \cdot \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2} = c \cdot \text{RMS}(x)
$$

Suy ra:

$$
\frac{x'}{\text{RMS}(x')} = \frac{cx}{c \cdot \text{RMS}(x)} = \frac{x}{\text{RMS}(x)}
$$

Do đó:

$$
\text{RMSNorm}(cx) = \text{RMSNorm}(x)
$$

Đây là tính chất vô cùng quan trọng (Scale Invariance - Bất biến với tỷ lệ). Mô hình sẽ trở nên ít nhạy cảm hơn với độ lớn tuyệt đối của activation, thay vào đó chủ yếu tập trung học hướng của vector biểu diễn.

---

## 7. So sánh với LayerNorm

### LayerNorm
$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
*Yêu cầu:* Mean, Variance, Scale ($\gamma$), Bias ($\beta$).

### RMSNorm
$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$
*Yêu cầu:* Mean Square, Scale ($\gamma$).

### Bảng so sánh trực quan

| Thành phần | LayerNorm | RMSNorm |
| :--- | :---: | :---: |
| **Mean ($\mu$)** | Có | Không |
| **Variance ($\sigma^2$)** | Có | Không |
| **Bias ($\beta$)** | Có | Không |
| **Scale Invariance** | Một phần | Tốt |
| **Chi phí tính toán** | Cao hơn | Thấp hơn |
| **LLM hiện đại** | Ít dùng | Rất phổ biến |

---

## 8. RMSNorm trong Transformer

Trong Transformer hiện đại, RMSNorm thường xuất hiện dưới cấu trúc **Pre-Norm** (chuẩn hóa trước khi đưa vào các khối tính toán chính).

### Khối Attention (Attention Block)
$$
y = x + \text{Attention}(\text{RMSNorm}(x))
$$

### Khối Feed Forward (Feed Forward Block)
$$
z = y + \text{MLP}(\text{RMSNorm}(y))
$$

Kiến trúc Pre-Norm này giúp:
* Khôi phục luồng gradient ổn định hơn qua kết nối tắt (Residual Connection).
* Hỗ trợ huấn luyện mô hình sâu hơn mà không sợ bão hòa.
* Giảm thiểu tối đa nguy cơ bùng nổ gradient (gradient explosion).

---

## 9. Vai trò trong Self-Attention

Trong cơ chế Self-Attention, các ma trận hình chiếu được tính toán:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Điểm số Attention ban đầu (Attention score):

$$
A = \frac{QK^T}{\sqrt{d}}
$$

Nếu độ lớn của ma trận đầu vào $X$ biến động quá mạnh ($\|X\| \rightarrow \text{không ổn định}$), tích vô hướng $QK^T$ cũng sẽ dao động cực đoan. Khi đó, hàm kích hoạt $\text{Softmax}$ có thể rơi vào vùng bão hòa, khiến phân phối trọng số quá nhọn hoặc quá phẳng, làm mô hình rất khó tối ưu.

RMSNorm đóng vai trò giữ cho $\|X\| \approx \text{constant}$ (hằng số), duy trì toán tử Attention luôn chạy trong vùng phân phối ổn định suốt quá trình huấn luyện.

---

## 10. Độ phức tạp tính toán

### LayerNorm
Do phải tính cả hai đại lượng độc lập là $\mu$ và $\sigma^2$, hệ thống cần thực hiện nhiều vòng lặp phép cộng tích lũy và tăng số lần truy cập đọc/ghi vào bộ nhớ (Memory I/O).
* Độ phức tạp tính toán: $\mathcal{O}(3d)$

### RMSNorm
Chỉ tính toán một đại lượng duy nhất là trung bình bình phương các phần tử $\sum x_i^2$, giúp giảm số phép toán và tối ưu luồng dữ liệu trên phần cứng GPU/TPU.
* Độ phức tạp tính toán: $\mathcal{O}(2d)$

Nhờ vậy, RMSNorm tiết kiệm được từ $10\%$ đến $50\%$ chi phí thời gian cho riêng thao tác cấu trúc Normalization khi cấu hình mô hình có kích thước chiều ẩn (Hidden size) lớn, chuỗi ngữ cảnh dài hoặc kiến trúc gồm hàng trăm layer.

---

## 11. Vai trò trong các LLM hiện đại

Các mô hình ngôn ngữ lớn hiện nay gần như đều chuyển sang sử dụng RMSNorm làm chuẩn mặc định thay cho LayerNorm truyền thống.

Lý do cốt lõi:
* Tính toán đơn giản, gọn nhẹ.
* Ít chiếm dụng băng thông bộ nhớ (Memory-bound bottleneck) nhờ lược bỏ toán tử toán học rườm rà.
* Tính chất bất biến tỷ lệ (Scale invariance) tốt hơn.
* Tương thích hoàn hảo với cấu trúc Pre-Norm giúp ổn định hệ thống ở quy mô hàng tỷ đến hàng nghìn tỷ tham số.

RMSNorm hiện là thành phần tiêu chuẩn cấu tạo nên:
* LLaMA (Meta)
* Mistral / Mixtral (Mistral AI)
* Gemma (Google)
* Qwen (Alibaba)
* DeepSeek
* Phi (Microsoft)

---

## 12. Tóm tắt

RMSNorm là phương pháp chuẩn hóa tối giản từ LayerNorm bằng cách loại bỏ hoàn toàn bước dịch tâm dữ liệu (mean-centering).

Công thức trung tâm:

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

Ý tưởng cốt lõi:
> Điều quan trọng để mạng nơ-ron sâu hội tụ ổn định không phải là việc đưa trung bình về 0, mà chính là khả năng kiểm soát chặt chẽ độ lớn (phạm vi biến thiên) của các vector biểu diễn.

Các đặc tính nổi bật:

* Giữ nguyên hướng vector
* Chuẩn hóa độ lớn activation
* Scale Invariance
* Chi phí thấp hơn LayerNorm
* Phù hợp với Transformer cực sâu
* Trở thành chuẩn mặc định trong hầu hết LLM hiện đại
