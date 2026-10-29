"""Seed cố định để tập bản đồ train/test giống nhau giữa mọi script (so sánh công bằng)."""

# Seed sinh tập bản đồ test (50 phòng) — agent không bao giờ được thấy các phòng này khi train.
TEST_ROOMS_SEED: int = 3000
# Seed sinh tập K bản đồ huấn luyện cố định (E4); loại trừ các phòng thuộc tập test.
TRAIN_ROOM_POOL_SEED: int = 2000
