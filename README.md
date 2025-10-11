# DeepFunGen Quick Start

DeepFunGen은 ONNX 포맷의 간섭도 예측 모델을 사용해 비디오를 큐에 넣고 백그라운드로 추론 및 후처리를 수행하는 Windows용 WinForms 앱입니다. 아래 단계만 따라 하면 바로 실행하고 결과를 확인할 수 있습니다.

## 준비물
- Windows 10/11 + .NET 8 런타임 (SDK 필요 없음)
- `app\bin\Debug\net8.0-windows\`에서 제공되는 `DeepFunGen.exe`와 동일 폴더의 DLL/모델 파일들
- `models/` 폴더에 추론에 사용할 ONNX 모델 여러 개(기본 `conv_tcn_5.onnx` 포함)

## 실행 방법
1. `DeepFunGen.exe`를 실행하면 기본 모델(`models/conv_tcn_5.onnx`)이 로드됩니다. 모델을 바꾸고 싶다면 우측 드롭다운에서 다른 ONNX 파일을 선택하세요.
2. 영상 파일(`.mp4` 등)을 메인 창으로 드래그하거나 `Add Video` 버튼을 눌러 추가하면 큐에 쌓이고 순서대로 처리됩니다.
3. 처리 진행 상황은 큐 목록의 상태 열과 하단 로그에서 확인할 수 있습니다. 추론이 끝나면:
   - `<영상이름>.<모델이름>.csv` 로 예측 결과가 저장되고,
   - 같은 폴더에 `<영상이름>.funscript` 파일이 생성됩니다.
4. 이미 예측된 영상을 다시 추가하면 저장된 CSV/Funscript를 불러와 재처리 시간을 절약합니다.
5. 큐에서 항목을 선택 후 `View Result`를 누르면 Prediction Viewer가 열려 단계별 그래프를 확인할 수 있습니다.

## 모델 교체 & 가속
- ONNX 모델은 `models/` 폴더에 넣으면 자동으로 드롭다운에 표시됩니다.
- GPU가 DirectML(DirectX 12)를 지원하면 자동으로 가속을 사용하며, 지원하지 않을 경우 CPU로 폴백됩니다. 실행 중 우측 상단의 Provider 라벨에서 현재 사용 중인 실행 프로바이더를 확인할 수 있습니다.

## 후처리 옵션
- 각 영상의 예측이 완료되면 기본 후처리 옵션으로 Funscript를 저장합니다. 글로벌 옵션은 `Options` 버튼에서 수정할 수 있으며, 변경 사항은 이후 작업에 적용됩니다.
