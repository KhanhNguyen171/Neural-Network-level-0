# Normalization in Deep Learning

## Từ BatchNorm đến GroupNorm và RMSNorm

---

## 1. Tại sao cần Normalization?

Trong mạng neural sâu, tín hiệu được truyền qua rất nhiều lớp:

$$
x \rightarrow f_1 \rightarrow f_2 \rightarrow \dots \rightarrow f_L
$$

Sau mỗi phép biến đổi tuyến tính $y = Wx + b$, độ lớn của activation có thể thay đổi đáng kể.

* Nếu activation quá lớn ($\|x\| \gg 1$), gradient dễ bùng nổ (Exploding Gradients).
* Nếu activation quá nhỏ ($\|x\| \ll 1$), gradient dễ biến mất (Vanishing Gradients).

Mục tiêu của Normalization là:
* Giữ activation ở một miền ổn định.
* Giảm độ nhạy với tỉ lệ scale của trọng số.
* Tăng tốc độ hội tụ toán học.
* Cải thiện khả năng tối ưu hóa của thuật toán gradient descent.

---

## 2. Batch Normalization (BN)

### Ý tưởng
BatchNorm thực hiện chuẩn hóa dữ liệu dọc theo chiều batch dữ liệu (mini-batch dimension).



Cho một tập mini-batch $B = \{x_1, x_2, \ldots, x_m\}$, quy trình tính toán gồm:

Tính giá trị trung bình (mean):
$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

Tính phương sai (variance):
$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

Chuẩn hóa vector:
$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Scale và Shift (Tạo điều kiện khôi phục biểu diễn gốc thông qua các tham số học được):
$$
y = \gamma \hat{x} + \beta
$$

### Tính chất
BatchNorm ép buộc phân phối của đầu ra trung gian đạt trạng thái lý tưởng:
$$
\mathbb{E}[\hat{x}] = 0 \quad \text{và} \quad \text{Var}[\hat{x}] = 1
$$

### Hạn chế
BatchNorm phụ thuộc rất lớn vào kích thước mini-batch (batch size). Khi $m \rightarrow \text{nhỏ}$, việc ước lượng $\mu_B$ và $\sigma_B^2$ bị nhiễu và không còn chính xác, khiến hiệu năng mô hình sụt giảm mạnh trong các tác vụ:
* Nhận diện vật thể (Object Detection)
* Phân vùng ảnh (Segmentation)
* Xử lý chuỗi Video (Video Models)
* Các mô hình cực lớn (Large Models) bị giới hạn phần cứng không thể đặt batch size lớn.

Đây chính là động lực lớn thúc đẩy sự ra đời của GroupNorm.

---

## 3. Layer Normalization (LN)

LayerNorm loại bỏ hoàn toàn sự phụ thuộc vào chiều Batch Dimension bằng cách chỉ chuẩn hóa dữ liệu cục bộ bên trong từng mẫu (sample).

Cho vector đặc trưng của một mẫu: $x = [x_1, x_2, \ldots, x_d]$.

Mean:
$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

Variance:
$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

Chuẩn hóa:
$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

### Đặc điểm
* Hoàn toàn độc lập với kích thước batch size.
* Hoạt động cực kỳ ổn định với mạng hồi quy RNN.
* Là nền tảng kiến trúc cốt lõi của mạng Self-Attention (Transformer).

---

## 4. Group Normalization (GN)

### Động cơ nghiên cứu
Nghiên cứu gốc của Group Normalization chỉ ra điểm chí mạng của BatchNorm khi huấn luyện với kích thước batch nhỏ. Thay vì chuẩn hóa trên toàn bộ các kênh (như LayerNorm) hay chuẩn hóa dọc theo Batch (như BatchNorm), GroupNorm chọn giải pháp trung hòa: Chia tập hợp các kênh đặc trưng thành nhiều nhóm nhỏ độc lập.

### Công thức
Giả sử Tensor đầu vào có dạng chuẩn trong thị giác máy tính:

$$
X \in \mathbb{R}^{N \times C \times H \times W}
$$

Hệ thống chia tổng số kênh $C$ thành $G$ nhóm (`groups`). Như vậy, mỗi nhóm đặc trưng sẽ sở hữu chính xác $\frac{C}{G}$ kênh.

Với từng nhóm kênh đặc trưng độc lập, hệ thống tính toán $\mu_g$ và $\sigma_g^2$ trên không gian không đổi của chiều cao ($H$) và chiều rộng ($W$):

$$
\mu_g = \frac{1}{m} \sum x_i
$$

$$
\sigma_g^2 = \frac{1}{m} \sum (x_i - \mu_g)^2
$$

Chuẩn hóa:
$$
\text{GN}(x) = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}
$$

### Các trường hợp đặc biệt
* Nếu cấu hình $G = 1$, GroupNorm biến đổi trở thành cấu trúc **LayerNorm (LN)**.
* Nếu cấu hình $G = C$ (mỗi nhóm gồm đúng 1 kênh), GroupNorm trở thành **InstanceNorm (IN)**.

Vì vậy, GN đóng vai trò là một cầu nối toán học tổng quát giữa LayerNorm và InstanceNorm. Do không phụ thuộc vào Batch Size, GN hoạt động vô cùng xuất sắc ngay cả khi thiết lập $\text{Batch} = 2$, trở thành sự lựa chọn ưu tiên trong các bài toán Downstream mạng CNN như Segmentation hay Object Detection.

---

## 5. LayerNorm có thực sự cần Mean?

Quy trình chuẩn hóa của LayerNorm thực chất gồm hai giai đoạn hình học:
1. **Mean Centering (Dịch tâm):** $x \rightarrow x - \mu$
2. **Variance Scaling (Co giãn phương sai):** $x \rightarrow \frac{x}{\sigma}$

Bài báo nghiên cứu về RMSNorm đã đặt ra một giả thuyết phản biện lớn: *Liệu bước dịch tâm (1) mang tính bất biến dịch chuyển (re-centering invariance) có thực sự đóng góp vào sự hội tụ, hay tất cả thành bại đều nằm ở bước co giãn độ lớn vector (2)?*

Các chứng minh thực nghiệm khẳng định rằng ta hoàn toàn có thể lược bỏ bước tính Mean mà không làm giảm độ chính xác của mạng nơ-ron.

---

## 6. RMSNorm

### Root Mean Square (Trị hiệu dụng)
Định nghĩa độ lớn vector thông qua trung bình bình phương:
$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

### Chuẩn hóa
RMSNorm giữ nguyên vị trí vector, chỉ thực hiện thao tác scale lại biên độ tổng thể:
$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

---

## 7. Ý nghĩa hình học

Xét chuẩn Euclidean ($L_2$ norm) của vector: $\|x\|_2 = \sqrt{\sum_i x_i^2}$. Ta có mối liên hệ mật thiết:
$$
\text{RMS}(x) = \frac{\|x\|_2}{\sqrt{d}}
$$

Do đó, ta có tỷ lệ thuận:
$$
\text{RMSNorm}(x) \propto \frac{x}{\|x\|_2}
$$

Về mặt hình học không gian, toán tử RMSNorm thực hiện phép chiếu đưa mọi vector biểu diễn về bề mặt của một siêu cầu (**hypersphere**) có bán kính cố định:
* **Giữ nguyên:** Hướng (góc quay) của vector.
* **Loại bỏ:** Sự nhiễu loạn về biên độ (độ dài) tuyệt đối của vector activation.

---

## 8. Tính chất Scale Invariance

Giả sử tín hiệu đầu vào bị phóng đại lên $c$ lần ($x' = cx$ với $c > 0$). Nhờ tính chất tuyến tính của căn bậc hai, ta có:
$$
\text{RMS}(x') = c \cdot \text{RMS}(x)
$$

Khi đưa vào phương trình chuẩn hóa:
$$
\frac{x'}{\text{RMS}(x')} = \frac{cx}{c \cdot \text{RMS}(x)} = \frac{x}{\text{RMS}(x)}
$$

Do đó:
$$
\text{RMSNorm}(cx) = \text{RMSNorm}(x)
$$

Đây chính là tính chất **Bất biến với tỷ lệ (Scale Invariance)**. Nó giúp hệ thống miễn nhiễm hoàn toàn với các cú sốc khuếch đại biên độ tín hiệu trong các mạng nơ-ron cực sâu.

---

## 9. Partial RMSNorm (pRMSNorm)

Để tối ưu hóa sâu hơn nữa cho các hệ thống phần cứng hiệu năng cao, bài báo RMSNorm đề xuất thêm một biến thể gọi là **pRMSNorm**.

Thay vì quét và tính toán trên toàn bộ $d$ chiều của vector, hệ thống chỉ trích xuất một phần tỷ lệ $p\%$ số lượng chiều đầu tiên để ước lượng giá trị RMS tổng thể:
$$
\text{RMS}_p(x) = \sqrt{\frac{1}{k} \sum_{i=1}^{k} x_i^2} \quad \text{với} \quad k = p\% \times d
$$

Giải pháp cải tiến này giúp:
* Giảm thiểu đáng kể thời gian xử lý toán tử trên GPU (Compute overhead).
* Giữ vững nguyên vẹn tính chất toán học quan trọng Scale Invariance.

---

## 10. So sánh các Normalization

| Method | Mean | Variance | Batch Dependent |
| :--- | :---: | :---: | :---: |
| **BatchNorm** | Dọc theo Batch | Dọc theo Batch | **Yes** |
| **LayerNorm** | Bên trong Mẫu | Bên trong Mẫu | No |
| **InstanceNorm** | Kênh của Mẫu | Kênh của Mẫu | No |
| **GroupNorm** | Nhóm Kênh của Mẫu | Nhóm Kênh của Mẫu | No |
| **RMSNorm** | *Không dùng* | *Không dùng (Dùng RMS)* | No |

---

## 11. Transformer và LLM hiện đại

Sự chuyển dịch kiến trúc qua các thế hệ mô hình ngôn ngữ:
* **Transformer gốc (2017) / BERT / GPT-2:** Trung thành với kiến trúc gốc sử dụng **LayerNorm**.
* **Các LLM thế hệ mới (LLaMA, Mistral, Gemma, Qwen, DeepSeek):** Đồng loạt chuyển đổi sang ứng dụng **RMSNorm**.

Nguyên nhân đến từ việc RMSNorm lược bỏ hoàn toàn các phép toán trừ dịch tâm và tham số bias, giúp giảm thiểu nghẽn băng thông bộ nhớ (Memory I/O bottleneck) của GPU, tăng tốc độ xử lý token/giây mà không làm suy giảm chất lượng perplexity của mô hình.

---

## 12. Kết luận

Lịch sử tiến hóa của các kỹ thuật Normalization trong học sâu là một chuỗi các cải tiến nhằm cắt bỏ dần các giả định toán học rườm rà, giải phóng năng lực tính toán cho phần cứng:

$$
\text{BatchNorm} \rightarrow \text{LayerNorm} \rightarrow \text{GroupNorm} \rightarrow \text{RMSNorm}
$$

* **BatchNorm:** Đặt nền móng giải quyết bài toán mất ổn định phân phối (Covariate Shift).
* **GroupNorm:** Phá bỏ xiềng xích phụ thuộc vào kích thước Batch dữ liệu.
* **LayerNorm:** Định hình chuẩn cấu trúc mã hóa cho cơ chế Attention.
* **RMSNorm:** Đạt đến sự tối giản tối đa khi chứng minh việc điều khiển biên độ vector quan trọng hơn việc tìm tâm dữ liệu.

---

## 13. Tổng quan hệ thống (Review)

### Normalization Family Overview

| Loại | Động cơ | Công thức cốt lõi (không gian $B, C, H, W$) | Đặc điểm |
| :--- | :--- | :--- | :--- |
| **BN (Batch Normalization)** | Giảm **Internal Covariate Shift**, tăng tốc độ hội tụ và ổn định huấn luyện. | Với mỗi kênh $c$:<br>$$\mu_c=\frac{1}{NHW}\sum x$$<br>$$\sigma_c^2=\frac{1}{NHW}\sum(x-\mu_c)^2$$<br>$$y=\gamma\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta$$ | - Hiệu quả rất cao với CNN.<br>- Tăng tốc độ học.<br>- Phụ thuộc nặng vào Batch Size.<br>- Kém hiệu quả nếu batch quá nhỏ. |
| **LN (Layer Normalization)** | Loại bỏ hoàn toàn sự phụ thuộc vào Batch Size, thích hợp cho dữ liệu tuần tự. | Với mỗi mẫu (sample):<br>$$\mu=\frac{1}{d}\sum_{i=1}^{d}x_i$$<br>$$\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$$<br>$$y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$ | - Không phụ thuộc cấu hình batch.<br>- Phù hợp tuyệt đối với RNN.<br>- Là nền tảng xương sống của Transformer. |
| **IN (Instance Normalization)** | Triệt tiêu sự khác biệt về phong cách kiểu dáng, thiết kế chuyên biệt cho xử lý ảnh. | Với từng mẫu và từng kênh riêng biệt:<br>$$\mu=\frac{1}{HW}\sum x$$<br>$$\sigma^2=\frac{1}{HW}\sum(x-\mu)^2$$<br>$$y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$ | - Chuẩn hóa độc lập từng bức ảnh.<br>- Loại bỏ các thuộc tính nhiễu tương phản.<br>- Ứng dụng mạnh trong Style Transfer. |
| **GN (Group Normalization)** | Khắc phục điểm yếu suy giảm độ chính xác của BN khi Batch Size nhỏ. | Chia tổng số kênh thành $G$ nhóm:<br>$$\mu_g=\frac{1}{m}\sum x_i$$<br>$$\sigma_g^2=\frac{1}{m}\sum(x_i-\mu_g)^2$$<br>$$y=\gamma\frac{x-\mu_g}{\sqrt{\sigma_g^2+\epsilon}}+\beta$$ | - Hoàn toàn không phụ thuộc Batch Size.<br>- Hoạt động mượt mà khi batch cực nhỏ.<br>- Cầu nối LN và IN.<br>- Chuẩn mực trong mạng Segmentation/Detection. |
| **AdaLN / FiLM** | Điều kiện hóa (Conditioning) quá trình chuẩn hóa dựa trên các vector thông tin ngoại cảnh. | $$y=\gamma(z)\odot \text{LN}(x)+\beta(z)$$ | - Các hệ số toán học biến thiên linh hoạt theo hàm của điều kiện $z$.<br>- Cực kỳ phổ biến trong Diffusion Models. |
| **RMSNorm** | Loại bỏ cơ chế toán học Mean-Centering, chỉ giữ lại tính chất Re-Scaling Invariance. | $$\text{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}$$<br>$$y=\gamma\odot\frac{x}{\text{RMS}(x)+\epsilon}$$ | - Lược bỏ Mean và Variance.<br>- Tốc độ xử lý phần cứng vượt trội.<br>- Mặc định trong các mô hình LLaMA, DeepSeek, Gemma. |
| **Pre-Norm** | Thay đổi vị trí đặt lớp Norm để bảo toàn luồng gradient cho mạng siêu sâu. | $$y=x+F(\text{Norm}(x))$$ | - Gradient truyền thẳng qua kết nối tắt ổn định.<br>- Thích hợp huấn luyện mô hình quy mô lớn.<br>- Tiêu chuẩn cấu tạo LLM ngày nay. |
| **Post-Norm** | Thiết kế nguyên bản của cấu trúc Transformer thế hệ đầu tiên. | $$y=\text{Norm}(x+F(x))$$ | - Kiến trúc trực quan truyền thống.<br>- Rất dễ gây bùng nổ hoặc mất mát tín hiệu khi tăng số layer. |
| **QK-Norm** | Ngăn chặn hiện tượng bão hòa hàm số mũ của toán tử Attention ở quy mô siêu lớn. | $$Q'=\frac{Q}{\|Q\|}, \quad K'=\frac{K}{\|K\|}$$<br>$$\text{Attention}=\text{Softmax}(Q'{K'}^T)$$ | - Kiểm soát chặt chẽ điểm số dot-product.<br>- Đảm bảo tính hội tụ ổn định cho các mạng ViT trên 22 tỷ tham số. |

---

### Mối quan hệ tiến hóa giữa các phương pháp

```text
BatchNorm
    │
    ├── InstanceNorm
    │
    ├── LayerNorm
    │      │
    │      ├── RMSNorm
    │      │
    │      └── AdaLN / FiLM
    │
    └── GroupNorm

---

## Evolution Timeline

```text
2015 ─ BatchNorm (Ioffe & Szegedy)
2016 ─ LayerNorm (Ba et al.)
2016 ─ InstanceNorm (Ulyanov et al.)
2018 ─ GroupNorm (He et al.)
2019 ─ RMSNorm (Zhang & Sennrich)
2022+ ─ AdaLN (DiT / Diffusion era)
2023+ ─ QK-Norm (ViT 22B / Giant Transformers)
```