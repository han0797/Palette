import os
import itertools
import pandas as pd
import numpy as np

CHANNELS = ["WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112", "IR123", "IR133"]

def get_channel_minmax(data_dir):

    ch_min_max = {}

    # 각 채널별 파일 읽기
    for ch in CHANNELS:
        file_path = os.path.join(data_dir, f"minmax_ami_{ch.lower()}.txt")  
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["datetime", "tcid", "min_value", "max_value"])
            min_val = df["min_value"].min()
            max_val = df["max_value"].max()
            ch_min_max[ch] = (np.floor(min_val), np.ceil(max_val))
        else:
            print(f"⚠ 파일 없음: {file_path}")

    # 채널 조합 생성 (중복 제거)
    #channel_pairs = list(itertools.combinations(CHANNELS, 2))
    channel_pairs = list(itertools.permutations(CHANNELS, 2))
    # 채널 차이 계산
    for ch1, ch2 in channel_pairs:
        min_diff = ch_min_max[ch1][0] - ch_min_max[ch2][1]  # 최소값 계산
        max_diff = ch_min_max[ch1][1] - ch_min_max[ch2][0]  # 최대값 계산
        ch_min_max[f"{ch1}-{ch2}"] = (np.floor(min_diff), np.ceil(max_diff))

    return ch_min_max
