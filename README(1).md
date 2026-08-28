# TurboWarp ONNX 얼굴 생성 확장

이 확장은 `face_diffusion.onnx`를 브라우저에서 실행해 32×32 얼굴 이미지를 만들고 현재 스프라이트에 표시합니다. 공식 Scratch가 아니라 **TurboWarp 전용**입니다.

## 1. GitHub 저장소 만들기

1. <https://github.com/new>에 접속합니다.
2. Repository name을 예를 들어 `scratch-diffusion`으로 입력합니다.
3. `Public`을 선택하고 `Create repository`를 누릅니다.
4. 저장소에서 `Add file` → `Upload files`를 선택합니다.
5. 다음 두 파일을 저장소 최상위에 업로드합니다.

```text
diffusion-extension.js
face_diffusion.onnx
```

6. 아래쪽의 `Commit changes`를 누릅니다.

ONNX 파일이 약 14MB이므로 GitHub의 일반 파일 업로드 한도 안에 들어갑니다.

## 2. GitHub Pages 켜기

1. 저장소 위쪽의 `Settings`를 누릅니다.
2. 왼쪽에서 `Pages`를 누릅니다.
3. `Build and deployment`의 Source를 `Deploy from a branch`로 선택합니다.
4. Branch를 `main`, 폴더를 `/ (root)`로 선택하고 `Save`를 누릅니다.
5. 배포가 끝날 때까지 몇 분 기다립니다.

주소는 다음 형태입니다.

```text
https://깃허브아이디.github.io/저장소이름/diffusion-extension.js
https://깃허브아이디.github.io/저장소이름/face_diffusion.onnx
```

예를 들어 GitHub 아이디가 `hoyeon`이고 저장소가 `scratch-diffusion`이면:

```text
https://hoyeon.github.io/scratch-diffusion/diffusion-extension.js
https://hoyeon.github.io/scratch-diffusion/face_diffusion.onnx
```

각 주소를 브라우저에서 열었을 때 404가 나오지 않아야 합니다.

## 3. TurboWarp에 확장 추가하기

1. <https://turbowarp.org/editor>를 엽니다.
2. 왼쪽 아래의 확장 기능 버튼을 누릅니다.
3. `Custom Extension` 또는 `사용자 정의 확장 프로그램`을 선택합니다.
4. `diffusion-extension.js`의 GitHub Pages 주소를 입력합니다.
5. **샌드박스 없이 실행**을 선택합니다.
6. 경고 내용을 확인한 다음 확장을 추가합니다.

## 4. 블록 사용 순서

초록 깃발을 클릭했을 때 아래 순서로 놓습니다.

```text
ONNX 모델 주소 [https://깃허브아이디.github.io/저장소이름/face_diffusion.onnx] 불러오기
[female] 얼굴 생성, 단계 [50]
생성 결과를 현재 스프라이트에 표시
```

- `female`: 여자 학습 폴더, 클래스 0
- `male`: 남자 학습 폴더, 클래스 1
- 단계 20: 빠르지만 결과가 거칠 수 있음
- 단계 50: 추천
- 단계 100: 더 느림
- `생성 진행률`: 0부터 100까지 표시
- `모델 상태`: 모델 로딩·생성·오류 상태 표시

생성 블록은 끝날 때까지 기다리는 블록입니다. 컴퓨터 성능에 따라 시간이 오래 걸릴 수 있습니다.

## 문제 해결

### 모델 입력 오류

ONNX 변환 시 입력과 출력 이름이 정확히 다음이어야 합니다.

```text
입력: image, timestep, label
출력: predicted_noise
```

### HTTPS 오류

GitHub의 일반 파일 주소인 `github.com/.../blob/...`가 아니라 `github.io`로 끝나는 GitHub Pages 주소를 사용합니다.

### 확장 프로그램이 보이지 않음

공식 Scratch 편집기가 아닌 TurboWarp를 사용했는지, 사용자 정의 확장을 샌드박스 없이 불러왔는지 확인합니다.

### 화면에 결과가 나오지 않음

`얼굴 생성` 블록이 완료된 뒤 `생성 결과를 현재 스프라이트에 표시` 블록을 실행해야 합니다. 무대가 아니라 표시할 스프라이트에서 블록을 실행하는 것이 좋습니다.
