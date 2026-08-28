(function (Scratch) {
  'use strict';

  if (!Scratch.extensions.unsandboxed) {
    throw new Error('이 확장은 TurboWarp에서 샌드박스 없이 실행해야 합니다.');
  }

  const SIZE = 32;
  const VALUES = 3 * SIZE * SIZE;
  const TRAIN_TIMESTEPS = 1000;
  const ORT_VERSION = '1.22.0';
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  let runtimePromise = null;

  const delay = () => new Promise(resolve => setTimeout(resolve, 0));
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  function errorMessage(error) {
    return error instanceof Error ? error.message : String(error);
  }

  function validateModelURL(rawURL) {
    let url;
    try {
      url = new URL(String(rawURL));
    } catch (_) {
      throw new Error('올바른 ONNX 모델 주소를 입력하세요.');
    }
    const local = url.hostname === 'localhost' || url.hostname === '127.0.0.1';
    if (url.protocol !== 'https:' && !(local && url.protocol === 'http:')) {
      throw new Error('인터넷의 ONNX 모델은 HTTPS 주소여야 합니다.');
    }
    return url.href;
  }

  function buildAlphaBars() {
    const result = new Float64Array(TRAIN_TIMESTEPS);
    let product = 1;
    for (let t = 0; t < TRAIN_TIMESTEPS; t++) {
      const beta = 0.0001 + (0.02 - 0.0001) * t / (TRAIN_TIMESTEPS - 1);
      product *= 1 - beta;
      result[t] = product;
    }
    return result;
  }

  function buildSamplingTimesteps(requested) {
    const numeric = Number(requested);
    const count = clamp(Number.isFinite(numeric) ? Math.round(numeric) : 50, 2, 100);
    const result = [];
    for (let i = 0; i < count; i++) {
      result.push(Math.round(999 * (count - 1 - i) / (count - 1)));
    }
    return result;
  }

  function gaussianNoise(length) {
    const values = new Float32Array(length);
    for (let i = 0; i < length; i += 2) {
      const u1 = Math.max(Math.random(), Number.EPSILON);
      const u2 = Math.random();
      const radius = Math.sqrt(-2 * Math.log(u1));
      values[i] = radius * Math.cos(2 * Math.PI * u2);
      if (i + 1 < length) values[i + 1] = radius * Math.sin(2 * Math.PI * u2);
    }
    return values;
  }

  function loadRuntime() {
    if (globalThis.ort) return Promise.resolve(globalThis.ort);
    if (runtimePromise) return runtimePromise;
    if (!globalThis.document || !document.createElement) {
      return Promise.reject(new Error('ONNX Runtime을 불러올 문서 환경이 없습니다.'));
    }

    runtimePromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = `${ORT_BASE}ort.min.js`;
      script.onload = () => {
        if (!globalThis.ort) {
          reject(new Error('ONNX Runtime이 로드되지 않았습니다.'));
          return;
        }
        globalThis.ort.env.wasm.wasmPaths = ORT_BASE;
        resolve(globalThis.ort);
      };
      script.onerror = () => reject(new Error('ONNX Runtime 다운로드에 실패했습니다.'));
      (document.head || document.documentElement).appendChild(script);
    });
    return runtimePromise;
  }

  class DiffusionFaces {
    constructor() {
      this.session = null;
      this.status = '모델을 불러오세요';
      this.progress = 0;
      this.generating = false;
      this.lastImage = null;
      this.lastSkinId = null;
      this.alphaBars = buildAlphaBars();
    }

    getInfo() {
      return {
        id: 'diffusionfaces',
        name: 'ONNX 얼굴 생성',
        color1: '#7354d8',
        color2: '#5d43b5',
        blocks: [
          {
            opcode: 'loadModel',
            blockType: Scratch.BlockType.COMMAND,
            text: 'ONNX 모델 주소 [URL] 불러오기',
            arguments: {URL: {type: Scratch.ArgumentType.STRING, defaultValue: 'https://USERNAME.github.io/REPOSITORY/face_diffusion.onnx'}}
          },
          {
            opcode: 'generate',
            blockType: Scratch.BlockType.COMMAND,
            text: '[GENDER] 얼굴 생성, 단계 [STEPS]',
            arguments: {
              GENDER: {type: Scratch.ArgumentType.STRING, menu: 'genders', defaultValue: 'female'},
              STEPS: {type: Scratch.ArgumentType.NUMBER, defaultValue: 50}
            }
          },
          {opcode: 'showOnSprite', blockType: Scratch.BlockType.COMMAND, text: '생성 결과를 현재 스프라이트에 표시'},
          {opcode: 'getProgress', blockType: Scratch.BlockType.REPORTER, text: '생성 진행률'},
          {opcode: 'getStatus', blockType: Scratch.BlockType.REPORTER, text: '모델 상태'}
        ],
        menus: {
          genders: {acceptReporters: true, items: ['female', 'male']}
        }
      };
    }

    getProgress() {
      return this.progress;
    }

    getStatus() {
      return this.status;
    }

    async loadModel(args) {
      try {
        const modelURL = validateModelURL(Scratch.Cast.toString(args.URL));
        this.status = '모델 불러오는 중';
        this.progress = 0;
        const ort = await loadRuntime();
        ort.env.wasm.wasmPaths = ORT_BASE;
        const session = await ort.InferenceSession.create(modelURL, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        for (const name of ['image', 'timestep', 'label']) {
          if (!session.inputNames.includes(name)) throw new Error(`모델 입력 ${name}이 없습니다.`);
        }
        if (!session.outputNames.includes('predicted_noise')) {
          throw new Error('모델 출력 predicted_noise가 없습니다.');
        }
        this.session = session;
        this.status = '준비 완료';
      } catch (error) {
        this.session = null;
        this.status = `오류: ${errorMessage(error)}`;
        throw error;
      }
    }

    async generate(args, util) {
      if (!this.session) throw new Error('먼저 ONNX 모델을 불러오세요.');
      if (this.generating) throw new Error('이미 얼굴을 생성하고 있습니다.');

      this.generating = true;
      this.status = '생성 중';
      this.progress = 0;
      try {
        const ort = globalThis.ort;
        const gender = Scratch.Cast.toString(args.GENDER).toLowerCase();
        const label = gender === 'male' ? 1n : 0n;
        const timesteps = buildSamplingTimesteps(Scratch.Cast.toNumber(args.STEPS));
        let sample = gaussianNoise(VALUES);

        // 생성 시작 시 랜덤 노이즈를 바로 보여준다.
        this.lastImage = sample;
        if (this._canDisplay(util)) this._renderImage(sample, util);
        await delay();

        for (let step = 0; step < timesteps.length; step++) {
          const t = timesteps[step];
          const previousT = step + 1 < timesteps.length ? timesteps[step + 1] : -1;
          const feeds = {
            image: new ort.Tensor('float32', sample, [1, 3, SIZE, SIZE]),
            timestep: new ort.Tensor('int64', BigInt64Array.from([BigInt(t)]), [1]),
            label: new ort.Tensor('int64', BigInt64Array.from([label]), [1])
          };
          const outputs = await this.session.run(feeds);
          const noise = outputs.predicted_noise && outputs.predicted_noise.data;
          if (!noise || noise.length !== VALUES) throw new Error('모델 출력 크기가 1×3×32×32가 아닙니다.');

          const alphaT = this.alphaBars[t];
          const alphaPrevious = previousT >= 0 ? this.alphaBars[previousT] : 1;
          const sqrtAlphaT = Math.sqrt(alphaT);
          const sqrtOneMinusAlphaT = Math.sqrt(1 - alphaT);
          const sqrtAlphaPrevious = Math.sqrt(alphaPrevious);
          const sqrtOneMinusAlphaPrevious = Math.sqrt(1 - alphaPrevious);
          const next = new Float32Array(VALUES);

          for (let i = 0; i < VALUES; i++) {
            if (!Number.isFinite(noise[i])) throw new Error('모델 출력에 잘못된 숫자가 있습니다.');
            const clean = clamp((sample[i] - sqrtOneMinusAlphaT * noise[i]) / sqrtAlphaT, -1, 1);
            next[i] = sqrtAlphaPrevious * clean + sqrtOneMinusAlphaPrevious * noise[i];
          }
          sample = next;
          this.progress = Math.round((step + 1) * 100 / timesteps.length);

          // 5단계마다, 그리고 마지막 단계에서 중간 결과를 갱신한다.
          if ((step + 1) % 5 === 0 || step === timesteps.length - 1) {
            this.lastImage = sample;
            if (this._canDisplay(util)) this._renderImage(sample, util);
          }
          await delay();
        }

        this.lastImage = sample;
        this.status = '생성 완료';
      } catch (error) {
        this.status = `오류: ${errorMessage(error)}`;
        throw error;
      } finally {
        this.generating = false;
      }
    }

    _canDisplay(util) {
      const renderer = Scratch.vm && Scratch.vm.renderer;
      return Boolean(
        renderer &&
        typeof renderer.createBitmapSkin === 'function' &&
        util &&
        util.target &&
        util.target.drawableID !== undefined &&
        globalThis.document &&
        document.createElement
      );
    }

    _renderImage(image, util) {
      const renderer = Scratch.vm.renderer;
      const source = document.createElement('canvas');
      source.width = SIZE;
      source.height = SIZE;
      const sourceContext = source.getContext('2d');
      const pixels = sourceContext.createImageData(SIZE, SIZE);
      const plane = SIZE * SIZE;
      for (let i = 0; i < plane; i++) {
        pixels.data[i * 4] = Math.round(clamp((image[i] + 1) * 127.5, 0, 255));
        pixels.data[i * 4 + 1] = Math.round(clamp((image[plane + i] + 1) * 127.5, 0, 255));
        pixels.data[i * 4 + 2] = Math.round(clamp((image[plane * 2 + i] + 1) * 127.5, 0, 255));
        pixels.data[i * 4 + 3] = 255;
      }
      sourceContext.putImageData(pixels, 0, 0);

      const display = document.createElement('canvas');
      display.width = 320;
      display.height = 320;
      const displayContext = display.getContext('2d');
      displayContext.imageSmoothingEnabled = false;
      displayContext.drawImage(source, 0, 0, display.width, display.height);

      const newSkinId = renderer.createBitmapSkin(display, 1);
      renderer.updateDrawableSkinId(util.target.drawableID, newSkinId);
      if (this.lastSkinId !== null && typeof renderer.destroySkin === 'function') {
        renderer.destroySkin(this.lastSkinId);
      }
      this.lastSkinId = newSkinId;
    }

    showOnSprite(args, util) {
      if (!this.lastImage) throw new Error('먼저 얼굴을 생성하세요.');
      const renderer = Scratch.vm && Scratch.vm.renderer;
      if (!renderer || typeof renderer.createBitmapSkin !== 'function') {
        throw new Error('TurboWarp 렌더러를 사용할 수 없습니다.');
      }
      if (!util || !util.target || util.target.drawableID === undefined) {
        throw new Error('표시할 스프라이트를 찾지 못했습니다.');
      }
      if (!globalThis.document || !document.createElement) throw new Error('캔버스를 만들 수 없습니다.');
      this._renderImage(this.lastImage, util);
    }
  }

  Scratch.extensions.register(new DiffusionFaces());
})(Scratch);
