# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

path_results = '/data-8t/khoon/rsrch_GMI5/Palette_claude/experiments/train_AMI2GMI_v0/'
path_log = f'{path_results}train.log'

def plot_training_log(log_path="train.log"):
    """train.log 파일을 읽어 CSI, CC, RMSE 지표 추이를 분석하고 그래프로 저장하는 함수"""

    if not os.path.exists(log_path):
        print(
            f">>> [Error] '{log_path}' 파일이 존재하지 않습니다. 경로를 확인해주세요."
        )
        return

    print(f">>> [System] Parsing log file: {log_path} ...")

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    epochs = []
    cc_vals = []
    rmse_vals = []
    csi_vals = []

    current_epoch = None
    lines = log_content.split("\n")

    # 정규식을 이용해 에폭 번호 및 검증 지표 정밀 추출
    for line in lines:
        epoch_match = re.search(r"epoch:\s*(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        if "val/cc:" in line:
            cc_vals.append(float(line.split(":")[-1].strip()))
        if "val/rmse:" in line:
            rmse_vals.append(float(line.split(":")[-1].strip()))
        if "val/csi:" in line:
            csi_vals.append(float(line.split(":")[-1].strip()))
            # 에폭 동기화 예외 처리
            if current_epoch is not None:
                epochs.append(current_epoch)
            else:
                epochs.append(len(csi_vals))

    # 데이터 길이 동기화 맞춤
    min_len = min(len(epochs), len(cc_vals), len(rmse_vals), len(csi_vals))
    df = pd.DataFrame(
        {
            "Epoch": epochs[:min_len],
            "CC": cc_vals[:min_len],
            "RMSE": rmse_vals[:min_len],
            "CSI": csi_vals[:min_len],
        }
    )

    # ---------------------------------------------------------------------
    # 시각화 플롯 구성 (이중 Y축 적용)
    # ---------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # 좌측 Y축 세팅 (CSI, CC)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("CSI / CC", fontsize=12, fontweight="bold", color="blue")
    (line1,) = ax1.plot(
        df["Epoch"],
        df["CSI"],
        label="Validation CSI",
        color="blue",
        alpha=0.8,
        linewidth=1.5,
    )
    (line2,) = ax1.plot(
        df["Epoch"],
        df["CC"],
        label="Validation CC",
        color="green",
        alpha=0.8,
        linewidth=1.5,
    )
    ax1.set_ylim(0, 0.7)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 우측 Y축 세팅 (RMSE)
    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE (K)", fontsize=12, fontweight="bold", color="red")
    (line3,) = ax2.plot(
        df["Epoch"],
        df["RMSE"],
        label="RMSE (K)",
        color="red",
        alpha=0.6,
        linestyle="--",
        linewidth=1.2,
    )
    ax2.set_ylim(15, 65)
    ax2.tick_params(axis="y", labelcolor="red")

    # 최고 기록 자동 탐색 및 하이라이트 지정
    best_idx = df["CSI"].idxmax()
    best_epoch = df.loc[best_idx, "Epoch"]
    best_csi = df.loc[best_idx, "CSI"]
    best_rmse = df.loc[best_idx, "RMSE"]

    # 최고 성능 시점에 세로 보라색 점선 및 어노테이션 박스 추가
    ax1.axvline(
        x=best_epoch, color="purple", linestyle=":", alpha=0.7, linewidth=2
    )
    ax1.annotate(
        f"Best Epoch {best_epoch}\n(CSI: {best_csi:.4f}\nRMSE: {best_rmse:.2f}K)",
        xy=(best_epoch, best_csi),
        xytext=(best_epoch - 60 if best_epoch > 100 else best_epoch + 20, best_csi + 0.1),
        arrowprops=dict(
            facecolor="purple", shrink=0.05, width=1, headwidth=6, alpha=0.7
        ),
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    # 범례 및 타이틀 결합
    lines_all = [line1, line2, line3]
    labels_all = [l.get_label() for l in lines_all]
    ax1.legend(lines_all, labels_all, loc="upper left", fontsize=10)

    plt.title(
        "Validation Metrics Trend (CSI, CC, RMSE)",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    plt.tight_layout()

    # 결과 저장물 분출
    output_img = f'{path_results}total_validation_trends.png'
    output_csv = f'{path_results}validation_metrics.csv'

    plt.savefig(output_img)
    df.to_csv(output_csv, index=False)

    print("-" * 60)
    print(f">>> [Success] 그래프 이미지 저장 완료: '{output_img}'")
    print(f">>> [Success] 정제 데이터 CSV 저장 완료: '{output_csv}'")
    print(
        f">>> [Best Milestone] 최적 에폭: {best_epoch} | 최고 CSI: {best_csi:.4f} | RMSE: {best_rmse:.2f}K"
    )
    print("-" * 60)


if __name__ == "__main__":
    # 실행할 로그 파일명을 여기에 넣어주세요
    plot_training_log(path_log)
