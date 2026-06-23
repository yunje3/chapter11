# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat


DATASET_NAME = "CWRU Bearing Dataset"
DATASET_URL = "https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets/data"
DEFAULT_FS = 12000
DEFAULT_MAX_SAMPLES = 250000


st.set_page_config(
    page_title="진동 데이터 상태진단 대시보드",
    layout="wide",
)


def find_signal_keys(mat_data):
    keys = []
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim <= 2 and arr.size > 100:
            keys.append(key)
    return sorted(keys)


def load_mat_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    return loadmat(uploaded_file)


@st.cache_data(show_spinner=False)
def load_mat_from_path(path):
    return loadmat(path)


@st.cache_data(show_spinner="KaggleHub에서 CWRU 샘플 데이터를 받는 중입니다...")
def download_cwru_dataset():
    import kagglehub

    return kagglehub.dataset_download("brjapon/cwru-bearing-datasets")


def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": rms,
        "peak": peak,
        "kurtosis": stats.kurtosis(signal, fisher=False),
        "skewness": stats.skew(signal),
        "crest_factor": peak / rms if rms > 0 else np.nan,
        "mean_abs": np.mean(np.abs(signal)),
    }


def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal)
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum


def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    signal = np.asarray(signal).ravel()
    window = max(1, int(fs * window_sec))
    step = max(1, int(fs * step_sec))
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        segment = signal[start : start + window]
        rows.append({"time_sec": start / fs, **calculate_features(segment)})
    return pd.DataFrame(rows)


def signal_chart(signal, fs, seconds):
    n = min(len(signal), int(fs * seconds))
    return pd.DataFrame(
        {
            "time_sec": np.arange(n) / fs,
            "amplitude": signal[:n],
        }
    ).set_index("time_sec")


def fft_chart(signal, fs, max_freq):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq
    return pd.DataFrame(
        {
            "frequency_hz": freq[mask],
            "amplitude": spectrum[mask],
        }
    ).set_index("frequency_hz")


def diagnose_windows(fault_win, rms_threshold, kurtosis_threshold, crest_threshold):
    def diagnose(row):
        reasons = []
        if row["rms"] > rms_threshold:
            reasons.append("RMS 증가")
        if row["kurtosis"] > kurtosis_threshold:
            reasons.append("Kurtosis 증가")
        if row["crest_factor"] > crest_threshold:
            reasons.append("Crest Factor 증가")

        if len(reasons) >= 2:
            return "위험", ", ".join(reasons)
        if len(reasons) == 1:
            return "주의", reasons[0]
        return "정상", "-"

    diagnosis = fault_win.copy()
    diagnosis[["diagnosis", "reason"]] = diagnosis.apply(
        lambda row: pd.Series(diagnose(row)),
        axis=1,
    )
    return diagnosis


def pick_default(paths, name_part):
    for path in paths:
        if name_part.lower() in Path(path).name.lower():
            return path
    return paths[0] if paths else None


def limit_signal(signal, max_samples):
    signal = np.asarray(signal).ravel()
    if len(signal) <= max_samples:
        return signal, False
    return signal[:max_samples], True


st.title("공개 진동 데이터 상태진단 대시보드")
st.caption(f"{DATASET_NAME} 기반 시간영역/주파수영역/특징값 분석")

with st.sidebar:
    st.header("데이터 설정")
    data_mode = st.radio(
        "데이터 입력 방식",
        ["MAT 파일 업로드", "저장소 raw 폴더", "KaggleHub 샘플 다운로드"],
    )
    fs = st.number_input("샘플링 주파수(Hz)", min_value=200, value=DEFAULT_FS, step=100)
    max_samples = st.number_input(
        "분석 최대 샘플 수",
        min_value=10000,
        value=DEFAULT_MAX_SAMPLES,
        step=10000,
        help="온라인 배포 환경에서 너무 큰 파일을 올렸을 때 앱이 느려지는 것을 막기 위한 제한입니다.",
    )
    seconds = st.slider("시간 파형 표시 길이(초)", 0.05, 2.0, 0.2, 0.05)
    max_freq = st.slider("FFT 최대 주파수(Hz)", 100, int(fs / 2), 1000, 100)
    window_sec = st.slider("윈도우 길이(초)", 0.05, 2.0, 0.2, 0.05)
    step_sec = st.slider("윈도우 이동 간격(초)", 0.05, 1.0, 0.1, 0.05)

normal_mat = None
fault_mat = None
normal_name = ""
fault_name = ""

if data_mode == "MAT 파일 업로드":
    col_upload_1, col_upload_2 = st.columns(2)
    with col_upload_1:
        normal_upload = st.file_uploader("정상 MAT 파일", type=["mat"])
    with col_upload_2:
        fault_upload = st.file_uploader("이상 MAT 파일", type=["mat"])

    normal_mat = load_mat_from_upload(normal_upload)
    fault_mat = load_mat_from_upload(fault_upload)
    normal_name = normal_upload.name if normal_upload else ""
    fault_name = fault_upload.name if fault_upload else ""
elif data_mode == "저장소 raw 폴더":
    raw_paths = sorted(str(path) for path in Path("raw").rglob("*.mat"))
    if not raw_paths:
        st.info("GitHub 저장소에 `raw/` 폴더를 만들고 `.mat` 파일을 넣으면 여기에서 선택할 수 있습니다.")
    else:
        col_repo_1, col_repo_2 = st.columns(2)
        default_normal = pick_default(raw_paths, "Time_Normal_1_098")
        default_fault = pick_default(raw_paths, "B007_1_123")
        with col_repo_1:
            normal_path = st.selectbox(
                "정상 MAT 파일",
                raw_paths,
                index=raw_paths.index(default_normal) if default_normal in raw_paths else 0,
            )
        with col_repo_2:
            fault_path = st.selectbox(
                "이상 MAT 파일",
                raw_paths,
                index=raw_paths.index(default_fault) if default_fault in raw_paths else 0,
            )

        normal_mat = load_mat_from_path(normal_path)
        fault_mat = load_mat_from_path(fault_path)
        normal_name = os.path.basename(normal_path)
        fault_name = os.path.basename(fault_path)
else:
    st.info("Streamlit Cloud에서는 첫 실행 시 데이터 다운로드에 시간이 걸릴 수 있습니다.")
    if st.button("CWRU 데이터 다운로드/불러오기", type="primary"):
        try:
            dataset_path = download_cwru_dataset()
            mat_paths = sorted(str(path) for path in Path(dataset_path).rglob("*.mat"))
            normal_path = pick_default(mat_paths, "Time_Normal_1_098")
            fault_path = pick_default(mat_paths, "B007_1_123")

            if normal_path and fault_path:
                normal_mat = load_mat_from_path(normal_path)
                fault_mat = load_mat_from_path(fault_path)
                normal_name = os.path.basename(normal_path)
                fault_name = os.path.basename(fault_path)
                st.session_state["sample_data"] = (normal_mat, fault_mat, normal_name, fault_name)
            else:
                st.error("샘플 MAT 파일을 찾지 못했습니다. 직접 파일 업로드 방식을 사용해 주세요.")
        except Exception as exc:
            st.error("KaggleHub 다운로드에 실패했습니다. MAT 파일 업로드 또는 raw 폴더 방식을 사용해 주세요.")
            st.caption(str(exc))

    if "sample_data" in st.session_state:
        normal_mat, fault_mat, normal_name, fault_name = st.session_state["sample_data"]

if normal_mat is None or fault_mat is None:
    st.warning("정상 파일과 이상 파일을 선택하면 분석 결과가 표시됩니다.")
    st.markdown(
        f"""
        - 데이터셋: [{DATASET_NAME}]({DATASET_URL})
        - 권장 파일 예시: `Time_Normal_1_098.mat`, `B007_1_123.mat`
        - CWRU 데이터는 보통 `X098_DE_time`, `X123_DE_time` 같은 변수명에 진동 신호가 들어 있습니다.
        - 온라인 배포에서 데이터를 항상 보여주고 싶다면 GitHub 저장소에 `raw/` 폴더를 만들고 샘플 `.mat` 파일 2개를 넣어 주세요.
        """
    )
    st.stop()

normal_keys = find_signal_keys(normal_mat)
fault_keys = find_signal_keys(fault_mat)

if not normal_keys or not fault_keys:
    st.error("MAT 파일에서 분석 가능한 1차원 신호 변수를 찾지 못했습니다.")
    st.stop()

col_key_1, col_key_2 = st.columns(2)
with col_key_1:
    normal_key = st.selectbox("정상 신호 변수", normal_keys, index=0)
with col_key_2:
    fault_key = st.selectbox("이상 신호 변수", fault_keys, index=0)

normal_signal = np.asarray(normal_mat[normal_key]).ravel()
fault_signal = np.asarray(fault_mat[fault_key]).ravel()
normal_original_len = len(normal_signal)
fault_original_len = len(fault_signal)
normal_signal, normal_trimmed = limit_signal(normal_signal, int(max_samples))
fault_signal, fault_trimmed = limit_signal(fault_signal, int(max_samples))

if normal_trimmed or fault_trimmed:
    st.info(
        f"온라인 실행 속도를 위해 신호를 앞부분 {int(max_samples):,}개 샘플로 제한했습니다. "
        "전체 분석이 필요하면 사이드바의 분석 최대 샘플 수를 늘려 주세요."
    )

st.subheader("데이터 개요")
overview_cols = st.columns(4)
overview_cols[0].metric("정상 파일", normal_name or "-")
overview_cols[1].metric("이상 파일", fault_name or "-")
overview_cols[2].metric("정상 샘플 수", f"{len(normal_signal):,}", f"원본 {normal_original_len:,}")
overview_cols[3].metric("이상 샘플 수", f"{len(fault_signal):,}", f"원본 {fault_original_len:,}")

feature_df = pd.DataFrame(
    [
        {"state": "normal", **calculate_features(normal_signal)},
        {"state": "fault", **calculate_features(fault_signal)},
    ]
)

tab_wave, tab_feature, tab_fft, tab_trend, tab_summary = st.tabs(
    ["시간 파형", "특징값", "FFT", "구간 추세", "진단 요약"]
)

with tab_wave:
    col_wave_1, col_wave_2 = st.columns(2)
    with col_wave_1:
        st.markdown("#### 정상 신호")
        st.line_chart(signal_chart(normal_signal, fs, seconds))
    with col_wave_2:
        st.markdown("#### 이상 신호")
        st.line_chart(signal_chart(fault_signal, fs, seconds))

with tab_feature:
    st.dataframe(feature_df, use_container_width=True)
    plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
    st.bar_chart(feature_df.set_index("state")[plot_cols])

with tab_fft:
    col_fft_1, col_fft_2 = st.columns(2)
    with col_fft_1:
        st.markdown("#### 정상 FFT")
        st.line_chart(fft_chart(normal_signal, fs, max_freq))
    with col_fft_2:
        st.markdown("#### 이상 FFT")
        st.line_chart(fft_chart(fault_signal, fs, max_freq))

normal_win = window_features(normal_signal, fs, window_sec, step_sec)
fault_win = window_features(fault_signal, fs, window_sec, step_sec)
normal_win["state"] = "normal"
fault_win["state"] = "fault"
trend_df = pd.concat([normal_win, fault_win], ignore_index=True)

normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])
rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
kurtosis_threshold = 5.0
crest_threshold = 4.0
diagnosis = diagnose_windows(fault_win, rms_threshold, kurtosis_threshold, crest_threshold)

with tab_trend:
    feature_to_plot = st.selectbox("추세로 볼 특징값", ["rms", "kurtosis", "crest_factor"])
    chart_df = trend_df.pivot_table(
        index="time_sec",
        columns="state",
        values=feature_to_plot,
        aggfunc="mean",
    )
    st.line_chart(chart_df)
    st.dataframe(trend_df.head(100), use_container_width=True)

with tab_summary:
    counts = diagnosis["diagnosis"].value_counts()
    col_diag_1, col_diag_2, col_diag_3 = st.columns(3)
    col_diag_1.metric("정상 구간", int(counts.get("정상", 0)))
    col_diag_2.metric("주의 구간", int(counts.get("주의", 0)))
    col_diag_3.metric("위험 구간", int(counts.get("위험", 0)))

    st.markdown("#### 진단 기준")
    st.write(
        pd.DataFrame(
            [
                {"feature": "rms", "threshold": rms_threshold, "rule": "정상 RMS 평균 + 3σ"},
                {"feature": "kurtosis", "threshold": kurtosis_threshold, "rule": "5 이상"},
                {"feature": "crest_factor", "threshold": crest_threshold, "rule": "4 이상"},
            ]
        )
    )

    st.markdown("#### 이상 신호 구간별 진단")
    st.dataframe(
        diagnosis[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]],
        use_container_width=True,
    )

    st.download_button(
        "진단 결과 CSV 다운로드",
        diagnosis.to_csv(index=False).encode("utf-8-sig"),
        file_name="vibration_diagnosis.csv",
        mime="text/csv",
    )
