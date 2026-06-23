# 진동 데이터 상태진단 Streamlit 대시보드

CWRU Bearing Dataset의 `.mat` 진동 데이터를 이용해 시간 파형, 특징값, FFT, 구간별 추세, 상태진단 결과를 확인하는 Streamlit 앱입니다.

## GitHub에 올릴 파일

아래 파일은 저장소에 꼭 포함합니다.

```text
app.py
requirements.txt
README.md
.streamlit/config.toml
.gitignore
```

선택 사항으로 샘플 데이터를 저장소에 같이 올리려면 아래처럼 `raw/` 폴더를 만들고 `.mat` 파일을 넣습니다.

```text
raw/
  Time_Normal_1_098.mat
  B007_1_123.mat
```

데이터 파일이 크면 GitHub 일반 업로드 대신 Git LFS를 사용하는 것이 좋습니다.

## Streamlit Community Cloud 배포

1. GitHub에서 새 저장소를 만듭니다.
2. 이 프로젝트 파일들을 저장소에 업로드합니다.
3. https://share.streamlit.io 에 로그인합니다.
4. `Create app` 또는 `New app`을 선택합니다.
5. Repository에 방금 만든 GitHub 저장소를 선택합니다.
6. Branch는 보통 `main`을 선택합니다.
7. Main file path에 `app.py`를 입력합니다.
8. Python version은 가능하면 로컬에서 테스트한 버전과 맞춥니다.
9. `Deploy`를 누릅니다.

## 앱에서 데이터 불러오기

- `MAT 파일 업로드`: 웹 화면에서 정상 `.mat` 파일과 이상 `.mat` 파일을 직접 업로드합니다.
- `저장소 raw 폴더`: GitHub 저장소의 `raw/` 폴더에 들어 있는 `.mat` 파일을 선택합니다.
- `KaggleHub 샘플 다운로드`: Streamlit 서버에서 CWRU 샘플 데이터를 다운로드해서 분석합니다.

CWRU 데이터는 보통 `X098_DE_time`, `X123_DE_time` 같은 변수명에 진동 신호가 들어 있습니다.

## 로컬 테스트

```bash
pip install -r requirements.txt
streamlit run app.py
```
