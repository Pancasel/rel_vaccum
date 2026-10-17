# Robot hút bụi phủ kín không gian bằng Học tăng cường (REL301 FA26)

Dạy robot hút bụi ảo tự học cách dọn hết căn phòng dạng lưới với ít bước nhất, bằng Q-learning
(phòng nhỏ) và Deep Q-learning (phòng lớn, phòng chưa từng thấy).

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
```

## Chạy test

```powershell
python -m pytest -m "not slow"
```

## Kết quả

(cập nhật ở T8.2)

## Tái lập thí nghiệm

(cập nhật ở T8.2)
