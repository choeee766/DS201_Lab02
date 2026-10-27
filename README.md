# DS201 Lab 2 — CNN trên VinaFood21

**Họ tên:** Phạm Ngọc Minh 

**MSSV:** 24521082 

**Lớp:** DS201.R12

Project này gồm bài giải và output đã chạy cho bốn bài của Lab02:
LeNet-5, GoogLeNet rút gọn, ResNet-18 và ResNet-50 pretrained từ Hugging Face.

## Cấu trúc

```text
DS201_Lab2_Project/
├── assignments.ipynb       Notebook Kaggle có code và kết quả đã chạy 
├── lab02_project/
│   ├── __init__.py          
│   ├── data.py              ImageFolder, transform, chia train/val/test
│   ├── model.py             Bốn kiến trúc CNN
│   ├── engine.py            train_one_epoch, evaluate, bốn độ đo
│   ├── train.py             Chạy thí nghiệm 
│   ├── evaluate.py          Đánh giá checkpoint 
│   └── requirements.txt     Các thư viện cần cài 
├── outputs/                 Kết quả chạy mới của các script
└── .gitignore
```

`assignments.ipynb` là **bài nộp chính**. File này ** chứa toàn bộ code cần chạy
trên Kaggle** và đã lưu output; notebook không import
`lab02_project`. Các file `.py` là cách tổ chức lại cùng phương pháp để có thể
chạy, kiểm tra và đọc từng phần qua terminal. Chúng không thay thế output đã
chạy trong notebook.

## Dữ liệu và phương pháp

Dữ liệu VinaFood21 có 21 lớp. Giữ nguyên tập `test/`; với tập `val/`,
tách 10% từ `train/` với seed 42. Train dùng augmentation;
val và test chỉ resize, chuyển tensor và chuẩn hoá ImageNet. Mọi
thí nghiệm dùng cùng index train/val. Checkpoint được chọn theo
val macro-F1, rồi mới đánh giá trên test

Notebook đã chạy với **9040 ảnh train, 1004 ảnh val, 6682 ảnh test**.
Kết quả test:

| Thí nghiệm | Accuracy | F1-macro |
|---|---:|---:|
| LeNet AvgPool | 0.3060 | 0.2883 |
| LeNet MaxPool | 0.3304 | 0.3137 |
| GoogLeNet rút gọn | 0.3831 | 0.3561 |
| ResNet-18 không augmentation | 0.4249 | 0.4092 |
| ResNet-18 có augmentation | 0.5079 | 0.4951 |
| ResNet-50 feature extraction | 0.7528 | 0.7608 |
| ResNet-50 fine-tuning | 0.8584 | 0.8591 |

Các độ đo còn lại, F1 từng lớp, confusion matrix và đường loss nằm trong
`assignments.ipynb`.

## Chạy notebook trên Kaggle

1. Tạo Kaggle Notebook từ `assignments.ipynb`.
2. Dùng **Add Input** gắn VinaFood21 sao cho có `train/` và `test/`.
3. Chọn GPU và bật Internet cho lần đầu tải `microsoft/resnet-50`.
4. Chạy từ đầu đến cuối nếu muốn huấn luyện lại. Notebook hiện đặt
   `RUN_EXPERIMENTS = True`; một lần Run All sẽ chạy đủ bảy thí nghiệm.

**Không cần upload thư mục `lab02_project` lên Kaggle để chạy notebook.** Chỉ
cần notebook và dataset VinaFood21. Các file đầu ra mới nằm trong
`/kaggle/working/output_lab2` theo cấu hình của notebook

## Chạy các module bằng terminal (tùy chọn)

Từ thư mục gốc project, cài `lab02_project/requirements.txt`, rồi dùng:

```bash
pip install -r lab02_project/requirements.txt
python -m lab02_project.train --model lenet_avg --data-root /path/to/VinaFood21
python -m lab02_project.train --model all --data-root /path/to/VinaFood21
python -m lab02_project.evaluate --checkpoint outputs/lenet_avg/best.pt --data-root /path/to/VinaFood21
```

`--model all` chạy bảy thí nghiệm; thời gian và kết quả có thể khác lần chạy
đã lưu trong notebook. ResNet-50 cần tải trọng số Hugging Face khi tạo mô hình.

