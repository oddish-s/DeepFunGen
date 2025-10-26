# DeepFunGen Quick Start

**DeepFunGen** is a Windows-based Python application that uses ONNX-format inference models to enqueue videos for background inference and post-processing.  
Follow the steps below to quickly set it up, run it, and view your results.

## Requirements
- Windows 10 or 11  
- [uv](https://github.com/astral-sh/uv)  
- One or more ONNX models inside the `models/` folder (the default model `conv_tcn_56.onnx` is included)

## How to Run
1. Run `run.bat`.  
   This will automatically install dependencies via **uv** and start the application.
2. **Add Files Screen**
   - By default, the model `models/conv_tcn_56.onnx` is selected at startup.  
     To switch to another model, choose a different ONNX file from the dropdown menu on the right.
   - Drag and drop your video files (e.g., `.mp4`) or click **Browse Files** to add them.
   - Adjust the options as needed, then click **Add to Queue** to enqueue the video for processing.
3. **Queue Screen**
   - Monitor processing progress here.
   - When inference is complete, prediction results are saved as `<video_name>.<model_name>.csv`, and a `<video_name>.funscript` file is generated in the same folder.
   - If you re-add a previously processed video, the saved CSV file will be reused to save reprocessing time.
   - Select an item in the queue and click **Open Viewer** to launch the Viewer and visualize the step-by-step graphs.

## Model Management & Acceleration
- Any ONNX models placed in the `models/` folder will automatically appear in the dropdown list.
- If your GPU supports **DirectML (DirectX 12)**, hardware acceleration will be enabled automatically; otherwise, the app will fall back to CPU execution.  
  You can check the current execution provider in the **Provider** section at the bottom.

---

_DeepFunGen aims to make ONNX-based functional signal generation and inference pipelines simple, efficient, and easy to visualize._


# DeepFunGen Quick Start

DeepFunGen은 ONNX 포맷의 간섭도 예측 모델을 사용해 비디오를 큐에 넣고 백그라운드로 추론 및 후처리를 수행하는 Windows용 python 앱입니다. 아래 단계만 따라 하면 바로 실행하고 결과를 확인할 수 있습니다.

## 준비물
- Windows 10/11
- uv (https://github.com/astral-sh/uv)
- `models/` 폴더에 추론에 사용할 ONNX 모델 여러 개(기본 `conv_tcn_56.onnx` 포함)

## 실행 방법
1. `run.bat`를 실행하면 uv가 종속성을 설치하고 프로그램이 시작됩니다
2. Add Files 화면
   - 시작시 기본 모델(`models/conv_tcn_56.onnx`)이 선택됩니다. 모델을 바꾸고 싶다면 우측 드롭다운에서 다른 ONNX 파일을 선택하세요.
   - 영상 파일(`.mp4` 등)을 드래그 앤 드랍하거나 `Browse Files` 버튼을 눌러 동영상을 추가할 수 있습니다.
   - 옵션을 조절한 다음 Add to queue 버튼을 대기열에 추가 할 수 있습니다.
3. Queue 화면
   - 처리 진행 상황을 확인할 수 있습니다.
   - 추론이 끝나면 `<영상이름>.<모델이름>.csv` 로 예측 결과가 저장되고, 같은 폴더에 `<영상이름>.funscript` 파일이 생성됩니다.
   - 이미 예측된 영상을 다시 추가하면 저장된 csv를 불러와 재처리 시간을 절약합니다.
   - 큐에서 항목을 선택 후 `Open Viewer`를 누르면 Viewer가 열려 단계별 그래프를 확인할 수 있습니다.

## 모델 교체 & 가속
- ONNX 모델은 `models/` 폴더에 넣으면 자동으로 드롭다운에 표시됩니다.
- GPU가 DirectML(DirectX 12)를 지원하면 자동으로 가속을 사용하며, 지원하지 않을 경우 CPU로 폴백됩니다. 하단의 Provider에서 현재 사용 중인 실행 프로바이더를 확인할 수 있습니다.
