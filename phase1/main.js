const startButton = document.querySelector("#startBtn");
const statusEl = document.querySelector("#status");
const video = document.querySelector("#webcam");
const overlay = document.querySelector("#overlay");
const mouthValueEl = document.querySelector("#mouthValue");
const leftEyeValueEl = document.querySelector("#leftEyeValue");
const rightEyeValueEl = document.querySelector("#rightEyeValue");
const avatarMount = document.querySelector("#avatar");

const EYE_LEFT_UPPER = 159;
const EYE_LEFT_LOWER = 145;
const EYE_RIGHT_UPPER = 386;
const EYE_RIGHT_LOWER = 374;
const MOUTH_UPPER = 13;
const MOUTH_LOWER = 14;
const SCALE_LEFT = 33;
const SCALE_RIGHT = 263;

let renderer;
let scene;
let camera;
let head;
let mouth;
let leftEye;
let rightEye;
let THREE;
let DrawingUtils;
let FaceLandmarker;
let FilesetResolver;
let drawingUtils;
let landmarker;
let stream;
let animationFrameId;
let running = false;
let avatarReady = false;

const channelState = {
  mouthOpen: 0,
  leftEyeOpen: 1,
  rightEyeOpen: 1,
};

function setStatus(message) {
  statusEl.textContent = message;
}

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function ema(previous, next, alpha = 0.2) {
  return previous + (next - previous) * alpha;
}

async function loadLibraries() {
  if (!THREE) {
    THREE = await import("https://unpkg.com/three@0.167.1/build/three.module.js");
  }

  if (!DrawingUtils || !FaceLandmarker || !FilesetResolver) {
    const mediapipe = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs"
    );
    DrawingUtils = mediapipe.DrawingUtils;
    FaceLandmarker = mediapipe.FaceLandmarker;
    FilesetResolver = mediapipe.FilesetResolver;
  }
}

async function createAvatar() {
  await loadLibraries();

  scene = new THREE.Scene();
  scene.background = new THREE.Color("#08111f");
  scene.fog = new THREE.Fog("#08111f", 6, 18);

  camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  camera.position.set(0, 0.2, 7.5);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  avatarMount.appendChild(renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 1.8);
  const key = new THREE.DirectionalLight(0x9bc7ff, 2.4);
  key.position.set(3, 6, 4);
  scene.add(ambient, key);

  const headMaterial = new THREE.MeshStandardMaterial({
    color: 0xf3d1c7,
    roughness: 0.65,
    metalness: 0.05,
  });
  head = new THREE.Mesh(new THREE.SphereGeometry(1.7, 48, 48), headMaterial);
  scene.add(head);

  const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0x111111 });
  leftEye = new THREE.Mesh(new THREE.SphereGeometry(0.15, 24, 24), eyeMaterial);
  rightEye = new THREE.Mesh(new THREE.SphereGeometry(0.15, 24, 24), eyeMaterial);
  leftEye.position.set(-0.52, 0.32, 1.35);
  rightEye.position.set(0.52, 0.32, 1.35);
  scene.add(leftEye, rightEye);

  const mouthMaterial = new THREE.MeshStandardMaterial({ color: 0x6b1d2a });
  mouth = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.18, 0.14), mouthMaterial);
  mouth.position.set(0, -0.62, 1.25);
  scene.add(mouth);

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.08, 0.4, 16),
    new THREE.MeshStandardMaterial({ color: 0xd9a691, roughness: 0.8 })
  );
  nose.rotation.x = Math.PI / 2;
  nose.position.set(0, -0.02, 1.34);
  scene.add(nose);

  const glow = new THREE.Mesh(
    new THREE.TorusGeometry(2.1, 0.03, 12, 100),
    new THREE.MeshBasicMaterial({ color: 0x7cc7ff })
  );
  glow.rotation.x = Math.PI / 2;
  glow.position.set(0, -0.05, -0.5);
  scene.add(glow);

  resizeAvatar();
  window.addEventListener("resize", resizeAvatar);
}

function resizeAvatar() {
  const width = avatarMount.clientWidth || 400;
  const height = avatarMount.clientHeight || 420;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function resizeOverlay() {
  const width = video.videoWidth || video.clientWidth || 640;
  const height = video.videoHeight || video.clientHeight || 480;
  overlay.width = width;
  overlay.height = height;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
}

function updateAvatar(channels) {
  const mouthScale = 0.55 + channels.mouthOpen * 3.2;
  const eyeScaleL = 0.45 + channels.leftEyeOpen * 2.2;
  const eyeScaleR = 0.45 + channels.rightEyeOpen * 2.2;

  mouth.scale.set(1.1, mouthScale, 1);
  mouth.position.y = -0.62 - channels.mouthOpen * 0.05;

  leftEye.scale.set(1, eyeScaleL, 1);
  rightEye.scale.set(1, eyeScaleR, 1);
}

function updateStats(channels) {
  mouthValueEl.textContent = channels.mouthOpen.toFixed(2);
  leftEyeValueEl.textContent = channels.leftEyeOpen.toFixed(2);
  rightEyeValueEl.textContent = channels.rightEyeOpen.toFixed(2);
}

function drawLandmarks(result) {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!result.faceLandmarks?.length) {
    return;
  }

  drawingUtils.drawLandmarks(result.faceLandmarks[0], {
    radius: 1.2,
    color: "#7cc7ff",
  });
}

function extractChannels(landmarks) {
  const scale = distance(landmarks[SCALE_LEFT], landmarks[SCALE_RIGHT]) || 1;
  const mouthOpen = clamp(distance(landmarks[MOUTH_UPPER], landmarks[MOUTH_LOWER]) / scale * 8);
  const leftEyeOpen = clamp(distance(landmarks[EYE_LEFT_UPPER], landmarks[EYE_LEFT_LOWER]) / scale * 14);
  const rightEyeOpen = clamp(distance(landmarks[EYE_RIGHT_UPPER], landmarks[EYE_RIGHT_LOWER]) / scale * 14);

  channelState.mouthOpen = ema(channelState.mouthOpen, mouthOpen, 0.22);
  channelState.leftEyeOpen = ema(channelState.leftEyeOpen, leftEyeOpen, 0.22);
  channelState.rightEyeOpen = ema(channelState.rightEyeOpen, rightEyeOpen, 0.22);

  return { ...channelState };
}

async function initLandmarker() {
  await loadLibraries();
  setStatus("Loading MediaPipe...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: new URL("../models/face_landmarker.task", window.location.href).href,
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

async function startCamera() {
  if (running) {
    return;
  }

  if (!avatarReady) {
    await createAvatar();
    avatarReady = true;
  }

  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  video.srcObject = stream;
  await video.play();
  resizeOverlay();

  if (!landmarker) {
    await initLandmarker();
  }

  drawingUtils = new DrawingUtils(overlay.getContext("2d"));
  running = true;
  setStatus("Running");
  renderLoop();
}

function renderLoop() {
  if (!running) {
    return;
  }

  const nowMs = performance.now();
  const result = landmarker.detectForVideo(video, nowMs);
  drawLandmarks(result);

  if (result.faceLandmarks?.length) {
    const channels = extractChannels(result.faceLandmarks[0]);
    updateAvatar(channels);
    updateStats(channels);
  }

  renderer.render(scene, camera);
  animationFrameId = requestAnimationFrame(renderLoop);
}

startButton.addEventListener("click", async () => {
  try {
    startButton.disabled = true;
    await startCamera();
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
    startButton.disabled = false;
    running = false;
  }
});

video.addEventListener("loadedmetadata", resizeOverlay);
setStatus("Ready");
