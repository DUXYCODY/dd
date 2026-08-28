const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const extensionPath = path.join(__dirname, '..', 'diffusion-extension.js');

function loadExtension(overrides = {}) {
  let registered;
  const Scratch = {
    extensions: {
      unsandboxed: true,
      register(value) { registered = value; }
    },
    BlockType: { COMMAND: 'command', REPORTER: 'reporter' },
    ArgumentType: { STRING: 'string', NUMBER: 'number' },
    Cast: {
      toString: String,
      toNumber: Number
    },
    vm: { renderer: overrides.renderer || null }
  };

  const context = {
    Scratch,
    URL,
    Float32Array,
    Uint8ClampedArray,
    BigInt64Array,
    Math,
    Number,
    Promise,
    Error,
    setTimeout,
    clearTimeout,
    console,
    globalThis: null,
    document: overrides.document,
    ort: overrides.ort
  };
  context.globalThis = context;
  vm.runInNewContext(fs.readFileSync(extensionPath, 'utf8'), context, {
    filename: extensionPath
  });
  return {extension: registered, context};
}

test('registers the user-visible extension blocks', () => {
  const {extension} = loadExtension();
  const info = extension.getInfo();
  assert.equal(info.id, 'diffusionfaces');
  assert.deepEqual(
    Array.from(info.blocks, block => block.opcode),
    ['loadModel', 'generate', 'showOnSprite', 'getProgress', 'getStatus']
  );
  assert.equal(extension.getProgress(), 0);
});

test('rejects insecure remote model URLs before loading a runtime', async () => {
  const {extension} = loadExtension();
  await assert.rejects(
    extension.loadModel({URL: 'http://example.com/face_diffusion.onnx'}),
    /HTTPS/
  );
});

function createOrt({inputNames, outputNames, noiseValue = 0} = {}) {
  const calls = [];
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }
  const session = {
    inputNames: inputNames || ['image', 'timestep', 'label'],
    outputNames: outputNames || ['predicted_noise'],
    async run(feeds) {
      calls.push(feeds);
      return {
        predicted_noise: {
          type: 'float32',
          dims: [1, 3, 32, 32],
          data: new Float32Array(3072).fill(noiseValue)
        }
      };
    }
  };
  return {
    env: {wasm: {}},
    Tensor,
    InferenceSession: {async create() { return session; }},
    calls,
    session
  };
}

test('loads the expected model interface and completes two DDIM steps', async () => {
  const ort = createOrt();
  const {extension} = loadExtension({ort});
  await extension.loadModel({URL: 'https://example.com/face_diffusion.onnx'});
  assert.equal(extension.getStatus(), '준비 완료');

  await extension.generate({GENDER: 'female', STEPS: 2});
  assert.equal(ort.calls.length, 2);
  assert.equal(ort.calls[0].label.data[0], 0n);
  assert.equal(ort.calls[0].timestep.data[0], 999n);
  assert.equal(ort.calls[1].timestep.data[0], 0n);
  assert.equal(extension.lastImage.length, 3072);
  assert.equal(Array.from(extension.lastImage).every(Number.isFinite), true);
  assert.equal(extension.getProgress(), 100);
  assert.equal(extension.getStatus(), '생성 완료');
});

test('rejects an ONNX model with incompatible input names', async () => {
  const ort = createOrt({inputNames: ['wrong'], outputNames: ['predicted_noise']});
  const {extension} = loadExtension({ort});
  await assert.rejects(
    extension.loadModel({URL: 'https://example.com/wrong.onnx'}),
    /모델 입력 image/
  );
  assert.match(extension.getStatus(), /^오류:/);
});

function createCanvasDocument() {
  const canvases = [];
  const document = {
    createElement(tag) {
      assert.equal(tag, 'canvas');
      const context = {
        imageSmoothingEnabled: true,
        createdImageData: null,
        drawnSource: null,
        createImageData(width, height) {
          this.createdImageData = {
            width,
            height,
            data: new Uint8ClampedArray(width * height * 4)
          };
          return this.createdImageData;
        },
        putImageData(imageData) { this.createdImageData = imageData; },
        drawImage(source) { this.drawnSource = source; }
      };
      const canvas = {
        width: 0,
        height: 0,
        context,
        getContext(kind) {
          assert.equal(kind, '2d');
          return context;
        }
      };
      canvases.push(canvas);
      return canvas;
    }
  };
  return {document, canvases};
}

test('converts CHW output to RGB and replaces its previous renderer skin', () => {
  const {document, canvases} = createCanvasDocument();
  const operations = [];
  let nextSkin = 10;
  const renderer = {
    createBitmapSkin(canvas, resolution) {
      operations.push(['create', canvas.width, canvas.height, resolution]);
      return nextSkin++;
    },
    updateDrawableSkinId(drawable, skin) { operations.push(['update', drawable, skin]); },
    destroySkin(skin) { operations.push(['destroy', skin]); }
  };
  const {extension} = loadExtension({document, renderer});
  extension.lastImage = new Float32Array(3072);
  extension.lastImage[0] = -1;
  extension.lastImage[1024] = 0;
  extension.lastImage[2048] = 1;

  extension.showOnSprite({}, {target: {drawableID: 7}});
  const rgba = canvases[0].context.createdImageData.data;
  assert.deepEqual(Array.from(rgba.slice(0, 4)), [0, 128, 255, 255]);
  extension.showOnSprite({}, {target: {drawableID: 7}});
  assert.deepEqual(operations, [
    ['create', 320, 320, 1],
    ['update', 7, 10],
    ['create', 320, 320, 1],
    ['update', 7, 11],
    ['destroy', 10]
  ]);
});
