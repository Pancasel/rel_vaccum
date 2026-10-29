"""Phần dùng chung của các script thí nghiệm.

Import gói này sẽ chuẩn bị tiến trình (cache matplotlib nằm trong dự án, console in được tiếng Việt)
TRƯỚC khi script import matplotlib/pandas, nên mỗi script thí nghiệm phải import gói này đầu tiên.
"""

from experiments.shared.console_setup import prepare_process_environment

prepare_process_environment()
