/* Iris Cortex — 3D cinematic visualization.
 *
 * Pure Three.js. Receives JSON events from Python over a QWebChannel
 * bridge (the `cortex` object registered in window.py).
 *
 * Event types handled (see docs/IRIS_VISUALIZATION.md → Data model):
 *   core.state, core.audio,
 *   node.spawn, node.activity, node.fade,
 *   edge.pulse, leaf.add.
 *
 * Camera: OrbitControls (drag = orbit, scroll = zoom).
 * Drag a node = grab + pull, releases back into orbit (spring).
 * Click a node = focus, emit ui.node_focused to Python.
 * Double-click a leaf = emit ui.leaf_opened.
 *
 * Author: Konstantin Markov
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { FilmPass } from 'three/addons/postprocessing/FilmPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ─── constants ─────────────────────────────────────────────────────

const COLORS = {
  bg: 0x050a14,
  core: 0xffffff,
  coreEmissive: { listening: 0x58e3ff, thinking: 0xf5b542, retrieving: 0xb388ff, speaking: 0xeaffff, idle: 0x2faecf, error: 0xef4444 },
  capability: 0x9fd3ff,
  project: 0x1de9b6,
  memory: 0xb388ff,
  tool: 0xf5b542,
  conversation: 0x58e3ff,
  leaf: 0xd6e8f5,
  // Unified light-sphere look for preview children (sub-sections,
  // branches, file leaves). All preview kids share a soft icy color
  // so the eye reads them as "small things orbiting their parent".
  file: 0xd6e8f5,
  subnode: 0xd6e8f5,
  edge: 0x4a8db0,
  edgeDim: 0x1d3a52,
  pulse: { cyan: 0x58e3ff, amber: 0xf5b542, magenta: 0xb388ff, red: 0xef4444 },
};

// Capability ring — staggered out of the equator so the layout
// reads as a 3D constellation rather than a flat dial.
const CAPABILITIES = [
  { id: 'cap-memory',   label: 'Memory',   angle: 0.0,            polar:  0.35 },
  { id: 'cap-voice',    label: 'Voice',    angle: Math.PI * 0.5,  polar: -0.30 },
  { id: 'cap-tools',    label: 'Tools',    angle: Math.PI,        polar:  0.32 },
  { id: 'cap-realtime', label: 'Realtime', angle: Math.PI * 1.5,  polar: -0.38 },
];

const CAP_RADIUS = 20;
const CONTEXT_RADIUS = 40;
const LEAF_ORBIT = 5;
const CORE_RADIUS = 5;        // smaller; sprite halo carries the glow

// ─── scene setup ───────────────────────────────────────────────────

const canvas = document.getElementById('cortex');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.setClearColor(COLORS.bg, 1);
// ACES filmic tone mapping gives HDR-like highlight roll-off so the
// core + bloom don't blow out — the scene reads as "filmed in space"
// rather than "WebGL diffuse colors."
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(COLORS.bg);
scene.fog = new THREE.FogExp2(COLORS.bg, 0.0028);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 18, 90);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.6;
controls.zoomSpeed = 0.8;
controls.minDistance = 14;
controls.maxDistance = 280;
controls.target.set(0, 0, 0);

// SUN LIGHTING — the Iris core is the only real light source in the
// scene (like a star). Planets (capabilities + projects) are
// directionally lit from the core, with a faint blue ambient so the
// shadow side doesn't go pitch black. Satellites pick up the sun's
// light too and read as small reflective bodies.
scene.add(new THREE.AmbientLight(0x162840, 0.10));
const sunLight = new THREE.PointLight(0xb6dcff, 2.6, 260, 1.3);
sunLight.position.set(0, 0, 0);
scene.add(sunLight);
// A tiny fill light on the opposite side keeps planets from going
// fully black on the far side — like Earth's earthshine.
const fillLight = new THREE.PointLight(0x46618c, 0.18, 180, 1.5);
fillLight.position.set(-30, -10, -50);
scene.add(fillLight);

// ─── post-processing ───────────────────────────────────────────────

const composer = new EffectComposer(renderer);
composer.setPixelRatio(renderer.getPixelRatio());
composer.setSize(window.innerWidth, window.innerHeight);
composer.addPass(new RenderPass(scene, camera));

// Bloom: strong cinematic glow, but selective. High threshold means
// only bright emissive elements (core, active nodes, edge pulses) bloom
// — background context nodes stay sharp. The previous "everything is
// blurred" look came from threshold=0.2 (dim things bloomed too); the
// fix is the threshold, not the strength.
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.70,   // strength — sprite halos provide most of the visible glow
  0.42,   // radius
  0.62,   // threshold — labels stay sharp; activity flares still bloom
);
composer.addPass(bloomPass);

const filmPass = new FilmPass(0.08, false);
composer.addPass(filmPass);

composer.addPass(new OutputPass());

// ─── starfield ─────────────────────────────────────────────────────

// ─── starfield (3 layers + twinkle) ────────────────────────────────
// Three distance shells with different sizes + colors give the void
// real depth. Each layer twinkles at a slightly different speed.
const starLayers = [];
function makeStars(count, rMin, rMax, size, color, opacity) {
  const positions = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const r = rMin + Math.random() * (rMax - rMin);
    const t = Math.random() * Math.PI * 2;
    const p = Math.acos(2 * Math.random() - 1);
    positions[i * 3]     = r * Math.sin(p) * Math.cos(t);
    positions[i * 3 + 1] = r * Math.sin(p) * Math.sin(t);
    positions[i * 3 + 2] = r * Math.cos(p);
    phases[i] = Math.random() * Math.PI * 2;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const m = new THREE.PointsMaterial({
    color, size, transparent: true, opacity,
    sizeAttenuation: true, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const pts = new THREE.Points(g, m);
  starLayers.push({ pts, mat: m, baseOpacity: opacity, twinkleSpeed: 0.4 + Math.random() * 0.6 });
  scene.add(pts);
}
makeStars(4500, 700, 950, 0.40, 0xcfdde9, 0.55);   // distant fine grain
makeStars(1800, 480, 680, 0.70, 0xb6cee6, 0.65);   // middle layer
makeStars( 500, 320, 470, 1.10, 0xe6f3ff, 0.80);   // closer brighter stars

// ─── nebula backdrop ───────────────────────────────────────────────
// A huge inside-rendered sphere with a soft radial gradient texture
// gives the scene a subtle "deep space cloud" feel without dominating
// the foreground. Rendered before everything else; depth-disabled.
(function makeNebula() {
  const RES = 512;
  const cv = document.createElement('canvas');
  cv.width = RES; cv.height = RES;
  const ctx = cv.getContext('2d');
  const grad = ctx.createRadialGradient(RES/2, RES/2, RES * 0.05, RES/2, RES/2, RES * 0.55);
  grad.addColorStop(0.00, 'rgba(40, 60, 100, 0.45)');
  grad.addColorStop(0.30, 'rgba(28, 36, 72, 0.30)');
  grad.addColorStop(0.65, 'rgba(20, 18, 40, 0.18)');
  grad.addColorStop(1.00, 'rgba(5, 10, 20, 0.00)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, RES, RES);
  // Sprinkle a few soft magenta/cyan smudges for color variation.
  for (let i = 0; i < 8; i++) {
    const cx = Math.random() * RES;
    const cy = Math.random() * RES;
    const radius = 60 + Math.random() * 90;
    const tint = Math.random() < 0.5
      ? 'rgba(118, 70, 160, 0.10)'   // magenta
      : 'rgba(60, 130, 180, 0.10)';  // cyan
    const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    g2.addColorStop(0, tint);
    g2.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, RES, RES);
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const geo = new THREE.SphereGeometry(1200, 32, 24);
  const mat = new THREE.MeshBasicMaterial({
    map: tex, side: THREE.BackSide,
    transparent: true, opacity: 0.85,
    depthWrite: false, fog: false,
    blending: THREE.AdditiveBlending,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.y = Math.random() * Math.PI;
  scene.add(mesh);
})();

// ─── drifting ambient particles ────────────────────────────────────
// Small dust specks moving slowly through the inner scene volume.
// Adds "flow" — even when nothing's happening the void breathes.
const ambientParticles = (() => {
  const N = 220;
  const positions = new Float32Array(N * 3);
  const velocities = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    // Spread through an inner volume around the constellation.
    positions[i * 3 + 0] = (Math.random() - 0.5) * 200;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 100;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
    velocities[i * 3 + 0] = (Math.random() - 0.5) * 0.6;
    velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.3;
    velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const m = new THREE.PointsMaterial({
    color: 0x6a8fb5, size: 0.42, transparent: true, opacity: 0.55,
    sizeAttenuation: true, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const pts = new THREE.Points(g, m);
  scene.add(pts);
  return { pts, geom: g, velocities, count: N };
})();

function updateAmbientParticles(dt) {
  const pos = ambientParticles.geom.attributes.position.array;
  const vel = ambientParticles.velocities;
  const N = ambientParticles.count;
  const BOUND = 140;
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    pos[i3]     += vel[i3]     * dt * 8;
    pos[i3 + 1] += vel[i3 + 1] * dt * 8;
    pos[i3 + 2] += vel[i3 + 2] * dt * 8;
    // Wrap so they keep drifting through the scene forever.
    if (pos[i3]     >  BOUND) pos[i3]     = -BOUND;
    if (pos[i3]     < -BOUND) pos[i3]     =  BOUND;
    if (pos[i3 + 1] >  BOUND * 0.5) pos[i3 + 1] = -BOUND * 0.5;
    if (pos[i3 + 1] < -BOUND * 0.5) pos[i3 + 1] =  BOUND * 0.5;
    if (pos[i3 + 2] >  BOUND) pos[i3 + 2] = -BOUND;
    if (pos[i3 + 2] < -BOUND) pos[i3 + 2] =  BOUND;
  }
  ambientParticles.geom.attributes.position.needsUpdate = true;
}

// ─── halo sprite (shared by core and every node) ───────────────────

// Shared white-alpha radial-gradient texture for halo sprites. Sprites
// tint via material.color so state changes can recolor instantly
// without rebuilding the canvas texture. One texture for all halos.
const _haloTexture = (() => {
  const RES = 128;
  const canvas = document.createElement('canvas');
  canvas.width = RES; canvas.height = RES;
  const ctx = canvas.getContext('2d');
  const cx = RES / 2;
  const grad = ctx.createRadialGradient(cx, cx, 0, cx, cx, cx);
  grad.addColorStop(0.00, 'rgba(255,255,255,0.85)');
  grad.addColorStop(0.18, 'rgba(255,255,255,0.42)');
  grad.addColorStop(0.50, 'rgba(255,255,255,0.10)');
  grad.addColorStop(1.00, 'rgba(255,255,255,0.00)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, RES, RES);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
})();

// Soft additive halo behind a node — same technique as the label
// sprites (canvas gradient on a billboarded plane). Decoupling the
// visible "glow" from the material's emissive lets the mesh stay
// defined and sharp at rest, while still feeling alive.
function makeHaloSprite(colorHex, size) {
  const mat = new THREE.SpriteMaterial({
    map: _haloTexture,
    color: colorHex,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(size, size, 1);
  return sprite;
}

// ─── core (the glowing center) ─────────────────────────────────────

const coreInner = new THREE.Mesh(
  new THREE.IcosahedronGeometry(CORE_RADIUS, 4),
  new THREE.MeshStandardMaterial({
    color: 0xd6efff,
    emissive: COLORS.coreEmissive.idle,
    emissiveIntensity: 0.85,
    roughness: 0.32,
    metalness: 0.55,
  }),
);
scene.add(coreInner);

// Soft sprite halo behind the core — carries most of the "atmosphere"
// glow at idle. Tracked base size so the per-frame breath multiplier
// keeps the right magnitude.
const _coreHaloBaseSize = CORE_RADIUS * 4.5;
const coreHalo = makeHaloSprite(COLORS.coreEmissive.idle, _coreHaloBaseSize);
scene.add(coreHalo);

let coreState = 'idle';
let coreIntensity = 0.4;
let coreAudio = 0.0;        // 0..1 RMS for audio reactivity
let coreAudioDecay = 0.0;   // smoothed value used for visuals

function setCoreState(state, intensity) {
  coreState = state;
  coreIntensity = Math.max(0.0, Math.min(1.0, intensity ?? 0.6));
  const target = COLORS.coreEmissive[state] ?? COLORS.coreEmissive.idle;
  coreInner.material.emissive.setHex(target);
  // Material emissive stays modest — the sprite halo carries most of
  // the visible glow now. Intensity boosts both so state changes still
  // read at a glance.
  coreInner.material.emissiveIntensity = 0.7 + coreIntensity * 0.8;
  coreHalo.material.color.setHex(target);
  // Sprite halo opacity scales with intensity (additive blending).
  coreHalo.material.opacity = 0.65 + coreIntensity * 0.30;
}

// ─── nodes (capabilities, context, leaves) ─────────────────────────

const nodeMap = new Map();   // id -> NodeRef
const edges = [];            // active edges (lines)
const pulses = [];           // active traveling pulses
const childCount = new Map();    // parent_id -> number of live children
const childrenByParent = new Map(); // parent_id -> Set<child_id> (lookup helper)
// Per-frame count of preview kids sharing each orbit center. Used by
// _computeTarget to auto-fit the satellite ring radius so more kids
// = larger ring (no overlap).
const orbitCountByCenter = new Map();
// Dynamic sibling-connect edges that appear between the direct
// children of the focused node — relating them to each other in
// addition to their parent.
const siblingEdges = [];

// ─── focus state machine ──────────────────────────────────────────
// Overview is the default: every parent node displays its children
// as tiny preview dots. Clicking a parent enters focus mode: camera
// flies in, that node's children scale up to full size + show labels,
// sibling top-level nodes fade. Esc returns to overview.

let focusedNodeId = null;
let cameraState = 'overview';        // overview | focusing | focused | unfocusing
let cameraAnimStart = 0;
const CAMERA_ANIM_MS = 1100;
let cameraStartPos = null;
let cameraStartTarget = null;
let cameraEndPos = null;
let cameraEndTarget = null;
const cameraRestPos = new THREE.Vector3(0, 18, 90);
const cameraRestTarget = new THREE.Vector3(0, 0, 0);

// Walk up the parent chain to the nearest non-preview ancestor.
// Returns the id of that ancestor (capability / project / null).
function topLevelAncestorId(node) {
  let cur = node.parentId ? nodeMap.get(node.parentId) : null;
  while (cur && cur.isPreview && cur.parentId) {
    cur = nodeMap.get(cur.parentId);
  }
  return cur ? cur.id : null;
}

// The id of the node that this preview kid should ORBIT around at
// the moment. Three regimes:
//   - In overview, preview kids of all depths orbit their TOP-LEVEL
//     ancestor (Marketing for both branches AND leaves), forming a
//     single satellite cloud around it instead of chains.
//   - When the kid is a direct child of the focused node, it orbits
//     the focused node so it expands cleanly around it.
//   - When the kid is a GRANDCHILD of the focused node (its parent is
//     a direct child of focus), it orbits its own immediate parent
//     (the branch), forming a sub-cluster — so each branch becomes a
//     mini-constellation belonging to it.
function orbitCenterId(node) {
  if (focusedNodeId !== null) {
    if (node.parentId === focusedNodeId) return focusedNodeId;
    const parent = nodeMap.get(node.parentId);
    if (parent && parent.parentId === focusedNodeId) {
      return node.parentId;
    }
  }
  return topLevelAncestorId(node);
}

// Rebuild the per-orbit-center kid count. Called once per frame
// before node updates so auto-fit radius reflects who's actually
// orbiting where right now (e.g. when focus changes, kids migrate
// from top-level to the focused node).
function recomputeOrbitCounts() {
  orbitCountByCenter.clear();
  for (const node of nodeMap.values()) {
    if (!node.isPreview || !node.parentId) continue;
    const cid = orbitCenterId(node);
    if (!cid) continue;
    orbitCountByCenter.set(cid, (orbitCountByCenter.get(cid) || 0) + 1);
  }
}

// Graded focus visibility: returns 0..1 for how prominently to show
// each node.
//
//   Overview (focusedNodeId === null):
//     Everything visible at 1.0 — the satellite cloud is dense.
//
//   Focused on a node:
//     - That node + its direct children — 1.0
//     - Its parent — 0.70 (kept as the "anchor")
//     - Its siblings — 0.55
//     - Other preview kids ORBITING THE SAME center (same top-level
//       ancestor) — 0.50, so the cloud stays dense and the user
//       doesn't lose the rest of the satellites they saw moments ago
//     - Preview kids of an unrelated top-level — 0 (hidden)
//     - Other top-levels — 0.16–0.20
function focusVisibility(node) {
  if (focusedNodeId === null) return 1.0;
  if (node.id === focusedNodeId) return 1.0;
  if (node.parentId === focusedNodeId) return 1.0;

  const focused = nodeMap.get(focusedNodeId);
  if (focused) {
    const focusedParent = focused.parentId;
    if (focusedParent && node.id === focusedParent) return 0.70;          // parent of focused
    if (focusedParent && node.parentId === focusedParent) return 0.55;    // siblings of focused
  }

  // Ambient preview kids: still visible at half opacity as long as
  // they share a top-level ancestor with the focused tree, so the
  // cloud doesn't collapse when you click in.
  if (node.isPreview) {
    const myTop = topLevelAncestorId(node);
    if (focused) {
      const focusedTop = focused.isCapability || !focused.parentId
        ? focused.id
        : topLevelAncestorId(focused);
      if (myTop && myTop === focusedTop) return 0.50;
    }
    return 0.0;
  }

  if (node.isCapability) return 0.16;
  if (!node.parentId) return 0.20;
  return 0.18;
}

function isInFocusedTree(node) {
  return focusVisibility(node) >= 0.95;
}

// Add cross-edges between every pair of direct children of `parentId`.
// For ≤ 8 siblings: complete graph (all pairs).
// For more: each kid links to its 3 nearest neighbors in 3D space —
// avoids a hairball when there are many leaves.
function addSiblingEdges(parentId) {
  const set = childrenByParent.get(parentId);
  if (!set || set.size < 2) return;
  const ids = [...set];
  const refs = ids.map(id => nodeMap.get(id)).filter(Boolean);
  if (refs.length < 2) return;
  const mat = new THREE.LineBasicMaterial({
    color: 0xb6dcff,
    transparent: true,
    opacity: 0.0,           // fade in via updateSiblingEdges
  });

  const addLine = (a, b) => {
    const m = mat.clone();
    const geom = new THREE.BufferGeometry()
      .setFromPoints([a.mesh.position.clone(), b.mesh.position.clone()]);
    const line = new THREE.Line(geom, m);
    line.userData = { aId: a.id, bId: b.id, born: performance.now() };
    scene.add(line);
    siblingEdges.push(line);
  };

  if (refs.length <= 8) {
    for (let i = 0; i < refs.length; i++) {
      for (let j = i + 1; j < refs.length; j++) {
        addLine(refs[i], refs[j]);
      }
    }
    return;
  }
  // Many siblings: nearest-3 graph.
  for (let i = 0; i < refs.length; i++) {
    const dists = [];
    for (let j = 0; j < refs.length; j++) {
      if (i === j) continue;
      dists.push({
        j,
        d2: refs[i].mesh.position.distanceToSquared(refs[j].mesh.position),
      });
    }
    dists.sort((x, y) => x.d2 - y.d2);
    for (let k = 0; k < Math.min(3, dists.length); k++) {
      const j = dists[k].j;
      if (j > i) addLine(refs[i], refs[j]); // dedupe — only emit i<j pairs
    }
  }
}

function clearSiblingEdges() {
  for (const line of siblingEdges) {
    scene.remove(line);
    line.geometry.dispose();
    line.material.dispose();
  }
  siblingEdges.length = 0;
}

function updateSiblingEdges(now) {
  for (const line of siblingEdges) {
    const { aId, bId, born } = line.userData;
    const a = nodeMap.get(aId);
    const b = nodeMap.get(bId);
    if (!a || !b) continue;
    const positions = line.geometry.attributes.position.array;
    positions[0] = a.mesh.position.x; positions[1] = a.mesh.position.y; positions[2] = a.mesh.position.z;
    positions[3] = b.mesh.position.x; positions[4] = b.mesh.position.y; positions[5] = b.mesh.position.z;
    line.geometry.attributes.position.needsUpdate = true;
    // Fade in over ~500 ms; track focusVis on both endpoints so they
    // dim when the camera retreats.
    const tIn = Math.min(1, (now - born) / 500);
    const visMin = Math.min(a.focusVis, b.focusVis);
    line.material.opacity = 0.30 * tIn * visMin;
  }
}

function focusNode(id) {
  const node = nodeMap.get(id);
  if (!node) return;
  const kids = childCount.get(id) || 0;
  if (kids === 0) {
    // No children to expand — fall through to legacy notify (Python
    // may show an info panel later).
    try { pythonBridge?.nodeFocused(id); } catch (_) {}
    return;
  }
  focusedNodeId = id;
  cameraState = 'focusing';
  cameraAnimStart = performance.now();
  cameraStartPos = camera.position.clone();
  cameraStartTarget = controls.target.clone();
  // Pull camera back from the focused node along the current view
  // direction, far enough to frame the whole subnode cluster. Distance
  // scales with how many kids are about to expand — more kids means
  // a bigger orbital ring, which means we need more room.
  const viewDir = new THREE.Vector3()
    .subVectors(camera.position, controls.target)
    .normalize();
  cameraEndTarget = node.mesh.position.clone();
  const kidCount = childCount.get(id) || 0;
  // Match _computeTarget's auto-fit + expand math, then leave margin
  // for the grandchild sub-clouds that ride on each direct child.
  const expectedKidR = Math.max(3.0, 1.6 + Math.sqrt(Math.max(1, kidCount)) * 0.65) + 10.0;
  // Estimate grandchild reach: pick the biggest sibling group as a
  // rough upper bound (don't over-zoom out for branches with 2 kids
  // when one branch has 20). 6 units is a safe per-branch envelope.
  const grandchildEnvelope = 6.0;
  const camDistance = Math.max(55, (expectedKidR + grandchildEnvelope) * 3.5);
  cameraEndPos = cameraEndTarget.clone().addScaledVector(viewDir, camDistance);
  controls.enabled = false;
  showBreadcrumb(node.label);
  // Add sibling-connect edges between every pair of direct children
  // of the focused node — relating them to each other in addition to
  // their parent. Cleaned up on unfocus.
  clearSiblingEdges();
  addSiblingEdges(id);
  try { pythonBridge?.nodeFocused(id); } catch (_) {}
}

function unfocusNode() {
  if (focusedNodeId === null && cameraState === 'overview') return;
  cameraState = 'unfocusing';
  cameraAnimStart = performance.now();
  cameraStartPos = camera.position.clone();
  cameraStartTarget = controls.target.clone();
  cameraEndPos = cameraRestPos.clone();
  cameraEndTarget = cameraRestTarget.clone();
  controls.enabled = false;
  hideBreadcrumb();
  focusedNodeId = null;
  // Sibling-connect web fades + clears as we leave focus.
  clearSiblingEdges();
  try { pythonBridge?.nodeUnfocused(); } catch (_) {}
}

function showBreadcrumb(label) {
  const el = document.getElementById('breadcrumb');
  const cur = document.getElementById('bc-current');
  if (el && cur) {
    cur.textContent = label;
    el.classList.add('visible');
  }
}

function hideBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  if (el) el.classList.remove('visible');
}

const NODE_GEOMS = {
  capability: new THREE.OctahedronGeometry(2.0, 0),
  project:    new THREE.SphereGeometry(2.4, 22, 18),
  memory:     new THREE.SphereGeometry(2.4, 22, 18),
  tool:       new THREE.SphereGeometry(2.4, 22, 18),
  conversation: new THREE.SphereGeometry(2.4, 22, 18),
  // Light spheres — preview-mode kids orbit their parent. Unified
  // geometry so capability subnodes, project branches, and file
  // leaves all read as the same "small light orb" species.
  leaf:       new THREE.SphereGeometry(0.85, 18, 14),
  file:       new THREE.SphereGeometry(0.85, 18, 14),
  subnode:    new THREE.SphereGeometry(0.95, 18, 14),
};

const EDGE_MAT = new THREE.LineBasicMaterial({
  color: COLORS.edgeDim, transparent: true, opacity: 0.45,
});

class NodeRef {
  constructor({ id, label, category, weight, parentId, preview, path }) {
    this.id = id;
    this.label = label;
    this.category = category;
    this.weight = Math.max(0.0, Math.min(1.0, weight ?? 0.5));
    this.parentId = parentId ?? null;
    this.isLeaf = category === 'leaf';
    this.isCapability = category === 'capability';
    // Preview: tiny constellation dot around its parent until the
    // parent is focused. Set by node.spawn events from the Python side
    // (capability sub-sections, project file leaves).
    this.isPreview = !!preview;
    this.path = path || null;           // file path for double-click open
    this.fading = false;
    this.fadeUntil = 0;
    this.bornAt = performance.now();
    this.activityUntil = 0;
    this.activityIntensity = 0;
    this.dragged = false;
    this.ttlUntil = 0;          // for leaves
    this.targetRadius = this._computeTargetRadius();
    // Per-node orbital phase so context nodes spread around the core.
    this.orbitAngle = Math.random() * Math.PI * 2;
    // Wider polar range (~±0.75 rad ≈ ±43°) so the layout reads as a
    // true 3D constellation; not just a flat dial spinning in y=0.
    this.orbitPolar = (Math.random() - 0.5) * 1.5;
    this.orbitSpeed = 0.03 + Math.random() * 0.05;
    // Per-node orbit-radius multiplier so satellite altitudes vary a
    // bit — looks more like a real satellite cloud than a thin ring.
    this.orbitRadiusMul = 0.85 + Math.random() * 0.45;
    // Per-mesh slow rotation gives objects parallax-y depth as they
    // drift through the scene. Tiny values so it never feels frenetic.
    this.spinX = (Math.random() - 0.5) * 0.20;
    this.spinY = (Math.random() - 0.5) * 0.30;
    this.spinZ = (Math.random() - 0.5) * 0.10;
    // Mode progress: 0 = preview (tiny, no label), 1 = full (normal).
    // Non-preview nodes start at 1; preview nodes start at 0 and lerp
    // toward 1 when their parent is focused.
    this.modeProgress = this.isPreview ? 0 : 1;
    // Per-frame fade factor (1 = visible, 0 = hidden) — driven by
    // focus state. Lerped each frame for smooth in/out.
    this.focusVis = 1;

    const geom = NODE_GEOMS[category] ?? NODE_GEOMS.project;
    const color = COLORS[category] ?? COLORS.project;
    // Sun-lit planet feel: very low emissive (the sun does the work),
    // moderate roughness for proper diffuse falloff, low metalness so
    // they read as planet-y solids rather than metal spheres. The
    // body's tint color carries the category identity, the sun lights
    // it like a real celestial body.
    const isPreviewKid = !!preview;
    this.material = new THREE.MeshStandardMaterial({
      color: isPreviewKid ? 0xc2d6e6 : 0xc8dded,
      emissive: color,
      emissiveIntensity: isPreviewKid
        ? 0.08
        : (this.isCapability ? 0.10 : (0.12 + this.weight * 0.12)),
      roughness: isPreviewKid ? 0.58 : 0.52,
      metalness: 0.28,
    });
    this.mesh = new THREE.Mesh(geom, this.material);
    this.mesh.scale.setScalar(0.0001);   // spawn animation starts tiny
    this.mesh.userData.nodeId = id;

    this.velocity = new THREE.Vector3();
    this.target = new THREE.Vector3();
    this._computeTarget();
    this.mesh.position.copy(this.target);

    scene.add(this.mesh);

    // Soft halo sprite behind the node — same look as the label
    // sprites' background glow, sized to the node's role.
    this.haloBaseSize =
      this.isLeaf      ? 3.6 :
      this.isCapability ? 7.5 :
      (9.0 + this.weight * 5.0);
    this.haloBaseOpacity =
      this.isLeaf      ? 0.55 :
      this.isCapability ? 0.50 :
      (0.55 + this.weight * 0.25);
    this.halo = makeHaloSprite(color, this.haloBaseSize);
    this.halo.material.opacity = this.haloBaseOpacity;
    this.halo.position.copy(this.mesh.position);
    scene.add(this.halo);

    // Floating sprite label (uses a 2D canvas texture).
    this.labelSprite = makeLabelSprite(label, color);
    this.labelSprite.position.copy(this.mesh.position);
    this.labelSprite.position.y += this._labelOffset();
    scene.add(this.labelSprite);
  }

  _computeTargetRadius() {
    if (this.isCapability) return CAP_RADIUS;
    if (this.isLeaf) return LEAF_ORBIT;
    return CONTEXT_RADIUS;
  }

  _labelOffset() {
    return this.isLeaf ? 1.6 : (this.isCapability ? 3.6 : 4.4);
  }

  _computeTarget() {
    if (this.isCapability) {
      // 3D capability ring: angle + polar (not stuck in y=0).
      const p = this.orbitPolar;
      this.target.set(
        Math.cos(this.orbitAngle) * Math.cos(p) * CAP_RADIUS,
        Math.sin(p) * CAP_RADIUS,
        Math.sin(this.orbitAngle) * Math.cos(p) * CAP_RADIUS,
      );
      return;
    }
    if (this.isPreview && this.parentId) {
      // Orbit center: top-level ancestor in overview, focused node
      // when this kid is a direct child of focus. See orbitCenterId.
      const centerId = orbitCenterId(this);
      const center = centerId ? nodeMap.get(centerId) : null;
      const base = center ? center.mesh.position : new THREE.Vector3();
      const centerGeomR = center
        ? (center.isCapability ? 2.0 : 2.4)
        : 1.5;
      const centerScale = center ? Math.max(center.mesh.scale.x, 0.5) : 1.0;
      const centerEffR = centerGeomR * centerScale;
      // EQUAL distance: every kid orbiting this center sits at the
      // same ring radius. AUTO-FIT: more kids → wider ring so they
      // don't overlap. Expansion adds extra radius when the center
      // is focused so labels have room AND so grandchild sub-clusters
      // sitting on top of each branch don't crowd the central node.
      const n = centerId ? (orbitCountByCenter.get(centerId) || 1) : 1;
      const minClearance = centerEffR * 1.35;
      const autoFit = 1.6 + Math.sqrt(n) * 0.65;
      const baseRing = Math.max(minClearance, autoFit);
      // Direct kids of focus get a big push outward so each one becomes
      // the center of its own visible mini-cluster of grandchildren.
      const expand = 10.0 * this.modeProgress;
      const r = baseRing + expand;
      this.target.set(
        base.x + Math.cos(this.orbitAngle) * Math.cos(this.orbitPolar) * r,
        base.y + Math.sin(this.orbitPolar) * r,
        base.z + Math.sin(this.orbitAngle) * Math.cos(this.orbitPolar) * r,
      );
      return;
    }
    if (this.isLeaf) {
      const parent = nodeMap.get(this.parentId);
      const base = parent ? parent.mesh.position : new THREE.Vector3();
      this.target.set(
        base.x + Math.cos(this.orbitAngle) * LEAF_ORBIT,
        base.y + this.orbitPolar * 2,
        base.z + Math.sin(this.orbitAngle) * LEAF_ORBIT,
      );
      return;
    }
    // Context: target floats on a sphere of CONTEXT_RADIUS.
    const r = CONTEXT_RADIUS;
    this.target.set(
      Math.cos(this.orbitAngle) * Math.cos(this.orbitPolar) * r,
      Math.sin(this.orbitPolar) * r,
      Math.sin(this.orbitAngle) * Math.cos(this.orbitPolar) * r,
    );
  }

  fade(ms) {
    this.fading = true;
    this.fadeUntil = performance.now() + ms;
  }

  pulseActivity(intensity, durationMs) {
    this.activityIntensity = Math.max(this.activityIntensity, intensity);
    this.activityUntil = performance.now() + durationMs;
  }

  setTtl(ms) {
    this.ttlUntil = performance.now() + ms;
  }

  update(dt, now) {
    // Drift orbital angle. Pauses for: dragged, currently-focused
    // node, AND parent of currently-focused (so the user's "anchor"
    // doesn't slide around while they're reading sub-content).
    const focusedNode = focusedNodeId !== null ? nodeMap.get(focusedNodeId) : null;
    const isFocusedAnchor = (
      focusedNodeId !== null && (
        this.id === focusedNodeId ||
        (focusedNode && this.id === focusedNode.parentId)
      )
    );
    if (!this.dragged && !isFocusedAnchor) {
      // Satellite-like orbits: preview kids move noticeably around
      // their parent so the eye reads them as orbiting bodies, not
      // just placed dots. Capabilities + top-level nodes drift slower
      // so the overall scene stays calm.
      const speedFactor = this.isPreview ? 3.2 : (this.isCapability ? 0.4 : 0.6);
      this.orbitAngle += dt * this.orbitSpeed * speedFactor;
    }

    // Per-mesh slow rotation for parallax-y 3D feel. Also paused on
    // the focused anchor so it sits steady while you read.
    if (!isFocusedAnchor) {
      this.mesh.rotation.x += dt * this.spinX;
      this.mesh.rotation.y += dt * this.spinY;
      this.mesh.rotation.z += dt * this.spinZ;
    }

    // ---- mode interpolation (preview <-> full) ----
    // Preview children are tiny dots at rest. When their parent is
    // the focused node, they lerp toward full size + labels.
    const targetMode = (this.isPreview && this.parentId === focusedNodeId) ? 1 : 0;
    if (this.isPreview) {
      this.modeProgress += (targetMode - this.modeProgress) * Math.min(1, dt * 4.5);
    }

    // ---- focus visibility (graded) ----
    const visTarget = focusVisibility(this);
    this.focusVis += (visTarget - this.focusVis) * Math.min(1, dt * 4.0);

    this._computeTarget();

    // Position update — two regimes:
    //   - Rigid follow (capabilities, preview kids, leaves): the
    //     position is copied straight from the target each frame, so
    //     these orbit their parent like satellites around Earth. No
    //     spring lag when the parent is dragged; the whole orbital
    //     constellation snaps along with it.
    //   - Spring physics (full-mode top-level nodes — projects /
    //     context): velocity-damped force toward the target, so the
    //     user can drag them and feel weight.
    const rigidFollow = this.isCapability || this.isPreview || this.isLeaf;
    if (rigidFollow) {
      if (!this.dragged) {
        this.mesh.position.copy(this.target);
        this.velocity.set(0, 0, 0);
      }
    } else if (!this.dragged) {
      const toTarget = this.target.clone().sub(this.mesh.position);
      const force = toTarget.multiplyScalar(3.2);
      this.velocity.addScaledVector(force, dt);
      this.velocity.multiplyScalar(0.86); // damping
      this.mesh.position.addScaledVector(this.velocity, dt);
    }

    // ---- scale ----
    // Preview scale: tiny dot at modeProgress=0, full at modeProgress=1.
    const previewScale = this.isPreview ? (0.22 + 0.78 * this.modeProgress) : 1.0;
    const age = now - this.bornAt;
    let baseScale;
    if (age < 800 && !this.fading) {
      const t = Math.min(age / 800, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      const overshoot = 1.0 + 0.08 * Math.sin(eased * Math.PI);
      baseScale = eased * overshoot;
    } else if (!this.fading) {
      const activityT0 = Math.max(0, (this.activityUntil - now) / 600);
      const breath = 1.0 + 0.04 * Math.sin(now * 0.0021 + this.orbitAngle * 5);
      const activity = 1.0 + 0.25 * activityT0 * this.activityIntensity;
      baseScale = breath * activity;
    } else {
      baseScale = 1.0;
    }
    this.mesh.scale.setScalar(baseScale * previewScale);

    // ---- emissive (material) ----
    if (this.activityUntil < now) this.activityIntensity = 0;
    const baseEmissive = this.isCapability ? 0.18 : (0.25 + this.weight * 0.20);
    const activityT = Math.max(0, (this.activityUntil - now) / 600);
    const activityBoost = this.activityIntensity * 1.6 * activityT;
    this.material.emissiveIntensity = (baseEmissive + activityBoost) * this.focusVis;
    // Opacity needs MeshStandardMaterial.transparent for fade to work.
    // Set it once here (idempotent) so siblings can actually fade.
    if (!this.material.transparent) {
      this.material.transparent = true;
    }
    this.material.opacity = this.focusVis;

    // ---- halo ----
    // Preview kids: NO glow at rest. Halo opacity + size scale with
    // modeProgress, so the satellite dots are just dark spheres in
    // overview, and only light up when their parent is expanded.
    // Non-preview nodes keep the full halo always.
    const haloPulse = 1.0 + activityT * this.activityIntensity * 0.45;
    const previewMix = this.isPreview ? this.modeProgress : 1.0;
    this.halo.scale.setScalar(this.haloBaseSize * haloPulse * previewMix);
    this.halo.material.opacity = (this.haloBaseOpacity * previewMix
      + activityT * this.activityIntensity * 0.35) * this.focusVis;
    this.halo.position.copy(this.mesh.position);

    // ---- label ----
    // Hidden when fully in preview; appears as it expands.
    const labelTarget = this.isPreview ? this.modeProgress : 1.0;
    this.labelSprite.material.opacity = 0.95 * labelTarget * this.focusVis;

    // ---- fade-out (lifecycle end) ----
    if (this.fading) {
      const t = (this.fadeUntil - now) / 1500;
      const s = Math.max(0, t);
      this.mesh.scale.setScalar(baseScale * previewScale * s);
      this.material.emissiveIntensity *= s;
      this.labelSprite.material.opacity *= s;
      this.halo.material.opacity *= s;
      this.halo.scale.setScalar(this.haloBaseSize * previewHaloFactor * s);
    }

    // Leaf TTL: auto-fade after expiry.
    if (this.isLeaf && this.ttlUntil > 0 && now > this.ttlUntil && !this.fading) {
      this.fade(1500);
    }

    // Label follows the node.
    this.labelSprite.position.copy(this.mesh.position);
    this.labelSprite.position.y += this._labelOffset();
  }

  dispose() {
    scene.remove(this.mesh);
    scene.remove(this.labelSprite);
    scene.remove(this.halo);
    this.material.dispose();
    if (this.labelSprite.material.map) this.labelSprite.material.map.dispose();
    this.labelSprite.material.dispose();
    // _haloTexture is shared across all halo sprites — don't dispose
    // it; only dispose the per-sprite material.
    this.halo.material.dispose();
  }
}

// ─── label sprites ─────────────────────────────────────────────────

function makeLabelSprite(text, colorHex) {
  // Flat sprite labels — no shadowBlur halo, muted off-white fill
  // that stays UNDER the bloom threshold so text never blooms.
  // Dark stroke provides legibility against the dark background.
  // The node color still tells the user what kind of thing they're
  // looking at via the mesh itself.
  void colorHex; // intentionally unused — labels share one neutral color
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 44px -apple-system, Segoe UI, Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // Dark stroke for outline legibility.
  ctx.lineWidth = 7;
  ctx.strokeStyle = 'rgba(5, 10, 20, 0.95)';
  ctx.strokeText(text, 256, 64);
  // Muted off-white fill (luminance ~0.55 — comfortably under the
  // 0.62 bloom threshold so text doesn't pick up bloom).
  ctx.fillStyle = 'rgba(150, 170, 188, 1.0)';
  ctx.fillText(text, 256, 64);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const mat = new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false, depthTest: true,
    opacity: 0.92,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(16, 4, 1);
  return sprite;
}

// ─── edges + pulses ────────────────────────────────────────────────

function addEdge(fromId, toId, opts = {}) {
  const from = fromId === 'core' ? null : nodeMap.get(fromId);
  const to = toId === 'core' ? null : nodeMap.get(toId);
  const a = from ? from.mesh.position : new THREE.Vector3();
  const b = to ? to.mesh.position : new THREE.Vector3();
  const geom = new THREE.BufferGeometry().setFromPoints([a.clone(), b.clone()]);
  const mat = EDGE_MAT.clone();
  if (opts.dim) {
    mat.color.setHex(COLORS.edgeDim);
    mat.opacity = 0.35;
  } else {
    mat.color.setHex(COLORS.edge);
    mat.opacity = 0.55;
  }
  const line = new THREE.Line(geom, mat);
  // preview=true => edge connects a preview-mode child; its opacity
  // tracks the child's focus visibility so it brightens with its child.
  line.userData = { fromId, toId, preview: !!opts.preview };
  scene.add(line);
  edges.push(line);
}

function updateEdges() {
  for (const line of edges) {
    const { fromId, toId, preview } = line.userData;
    const fromNode = fromId === 'core' ? null : nodeMap.get(fromId);
    const toNode = toId === 'core' ? null : nodeMap.get(toId);
    const from = fromId === 'core' ? new THREE.Vector3() : fromNode?.mesh.position;
    const to = toId === 'core' ? new THREE.Vector3() : toNode?.mesh.position;
    if (!from || !to) continue;
    const positions = line.geometry.attributes.position.array;
    positions[0] = from.x; positions[1] = from.y; positions[2] = from.z;
    positions[3] = to.x;   positions[4] = to.y;   positions[5] = to.z;
    line.geometry.attributes.position.needsUpdate = true;

    // Preview edges show ONLY relationships currently being expressed:
    //   - kid's parent IS the focused node (direct expansion edge)
    //   - kid's GRANDPARENT is the focused node (the kid is a leaf in
    //     a branch sub-cluster; the edge shows it belongs to that
    //     branch, NOT a far-away starburst)
    // Otherwise: hidden.
    let opacity;
    if (preview) {
      const kidNode = toNode;
      let visKind = 0;
      if (kidNode && kidNode.parentId === focusedNodeId) {
        visKind = 1;        // direct expansion — brighter
      } else if (kidNode && kidNode.parentId) {
        const parentNode = nodeMap.get(kidNode.parentId);
        if (parentNode && parentNode.parentId === focusedNodeId) {
          visKind = 2;      // grandchild → branch — dimmer
        }
      }
      opacity = visKind === 1 ? 0.40 : visKind === 2 ? 0.22 : 0.0;
    } else {
      const fromVis = fromId === 'core' ? 1.0 : (fromNode?.focusVis ?? 1.0);
      const toVis = toId === 'core' ? 1.0 : (toNode?.focusVis ?? 1.0);
      opacity = 0.55 * Math.min(fromVis, toVis);
    }
    line.material.opacity = opacity;
  }
}

function pulseEdge(fromId, toId, color, durationMs) {
  const colorHex = COLORS.pulse[color] ?? COLORS.pulse.cyan;
  const fromPos = fromId === 'core' ? new THREE.Vector3() : nodeMap.get(fromId)?.mesh.position;
  const toPos = toId === 'core' ? new THREE.Vector3() : nodeMap.get(toId)?.mesh.position;
  if (!fromPos || !toPos) return;
  const mat = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.55, 14, 12), mat);
  mesh.position.copy(fromPos);
  scene.add(mesh);
  pulses.push({
    mesh,
    fromId,
    toId,
    startAt: performance.now(),
    duration: durationMs,
    material: mat,
  });
}

function updatePulses(now) {
  for (let i = pulses.length - 1; i >= 0; i--) {
    const p = pulses[i];
    const t = (now - p.startAt) / p.duration;
    if (t >= 1) {
      scene.remove(p.mesh);
      p.material.dispose();
      p.mesh.geometry.dispose();
      pulses.splice(i, 1);
      continue;
    }
    const fromPos = p.fromId === 'core' ? new THREE.Vector3() : nodeMap.get(p.fromId)?.mesh.position;
    const toPos = p.toId === 'core' ? new THREE.Vector3() : nodeMap.get(p.toId)?.mesh.position;
    if (!fromPos || !toPos) continue;
    p.mesh.position.lerpVectors(fromPos, toPos, t);
    // Fade out near the end so the disappearance reads smooth.
    p.material.opacity = t < 0.85 ? 1.0 : (1.0 - (t - 0.85) / 0.15);
  }
}

// ─── seed capability ring ──────────────────────────────────────────

for (const cap of CAPABILITIES) {
  const ref = new NodeRef({
    id: cap.id,
    label: cap.label,
    category: 'capability',
    weight: 0.6,
    parentId: null,
  });
  ref.orbitAngle = cap.angle;
  ref.orbitPolar = cap.polar;
  nodeMap.set(cap.id, ref);
  addEdge('core', cap.id, { dim: true });
}

// ─── event handling (from Python via QWebChannel) ──────────────────

let evtCount = 0;
let evtWindowStart = performance.now();
let evtRate = 0;

function applyEvent(ev) {
  evtCount++;
  switch (ev.type) {
    case 'core.state':
      setCoreState(ev.state, ev.intensity);
      break;
    case 'core.audio':
      coreAudio = Math.max(0, Math.min(1, ev.rms ?? 0));
      break;
    case 'node.spawn': {
      if (nodeMap.has(ev.id)) break;
      const ref = new NodeRef({
        id: ev.id,
        label: ev.label,
        category: ev.category,
        weight: ev.weight,
        parentId: ev.parent_id,
        preview: ev.preview,
        path: ev.path,
      });
      nodeMap.set(ev.id, ref);
      // Track children-per-parent so the click handler can decide
      // whether a node is focusable (i.e. has something to expand).
      const parentForCount = ev.parent_id ?? 'core';
      childCount.set(parentForCount, (childCount.get(parentForCount) || 0) + 1);
      if (!childrenByParent.has(parentForCount)) {
        childrenByParent.set(parentForCount, new Set());
      }
      childrenByParent.get(parentForCount).add(ev.id);
      // Edge to parent (or core). Preview edges are tagged so
      // updateEdges() can fade them with the child's focus visibility.
      addEdge(ev.parent_id ?? 'core', ev.id, {
        dim: !!ev.preview,
        preview: !!ev.preview,
      });
      break;
    }
    case 'node.activity': {
      const ref = nodeMap.get(ev.id);
      if (ref) ref.pulseActivity(ev.intensity ?? 0.8, ev.duration_ms ?? 600);
      break;
    }
    case 'edge.pulse':
      pulseEdge(ev.from, ev.to, ev.color, ev.duration_ms ?? 500);
      break;
    case 'leaf.add': {
      const leafId = `${ev.parent_id}::leaf::${Math.random().toString(36).slice(2, 8)}`;
      const ref = new NodeRef({
        id: leafId,
        label: ev.label,
        category: 'leaf',
        weight: 0.4,
        parentId: ev.parent_id,
      });
      nodeMap.set(leafId, ref);
      ref.setTtl(ev.ttl_ms ?? 8000);
      addEdge(ev.parent_id, leafId);
      break;
    }
    case 'node.fade': {
      const ref = nodeMap.get(ev.id);
      if (ref) ref.fade(ev.duration_ms ?? 1500);
      break;
    }
    default:
      break;
  }
}

// ─── QWebChannel bridge ────────────────────────────────────────────

let pythonBridge = null;

function initBridge() {
  if (typeof QWebChannel === 'undefined') {
    console.warn('[cortex] QWebChannel shim missing — running standalone');
    return;
  }
  new QWebChannel(qt.webChannelTransport, (channel) => {
    pythonBridge = channel.objects.cortex;
    if (!pythonBridge) {
      console.warn('[cortex] No "cortex" object on channel');
      return;
    }
    pythonBridge.event.connect((payloadJson) => {
      try {
        applyEvent(JSON.parse(payloadJson));
      } catch (e) {
        console.error('[cortex] bad event payload', e, payloadJson);
      }
    });
    // Tell Python the JS side is wired up.
    try { pythonBridge.jsReady(); } catch (e) { /* harmless if not present */ }
  });
}
initBridge();

// ─── mouse interaction (hover, drag, click) ────────────────────────

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredId = null;
let draggingRef = null;
let dragPlane = new THREE.Plane();
let dragOffset = new THREE.Vector3();
let dragLast = new THREE.Vector3();
let pressTime = 0;
const hoverLabel = document.getElementById('hover-label');

function pickNodeUnderMouse() {
  const meshes = [];
  for (const ref of nodeMap.values()) {
    if (!ref.fading) meshes.push(ref.mesh);
  }
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(meshes, false);
  return hits.length ? hits[0] : null;
}

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  if (draggingRef) {
    // Project mouse onto the drag plane.
    raycaster.setFromCamera(mouse, camera);
    const newPoint = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(dragPlane, newPoint)) {
      newPoint.sub(dragOffset);
      // Bound the drag so the node can't fly off into deep space when
      // the mouse drags close to the screen edge (where the ray is
      // grazing the drag plane and the intersection is far away).
      // Clamp to a soft bubble around the node's orbital target.
      const target = draggingRef.target;
      const maxDragR = 40;
      const offset = new THREE.Vector3().subVectors(newPoint, target);
      if (offset.length() > maxDragR) {
        offset.setLength(maxDragR);
        newPoint.copy(target).add(offset);
      }
      // And hard-cap distance from the core so the cortex stays whole.
      if (newPoint.length() > 120) newPoint.setLength(120);
      draggingRef.mesh.position.copy(newPoint);
      draggingRef.velocity.set(0, 0, 0);
      dragLast.copy(newPoint);
    }
    return;
  }

  const hit = pickNodeUnderMouse();
  const id = hit ? hit.object.userData.nodeId : null;
  if (id !== hoveredId) {
    hoveredId = id;
    if (id) {
      const ref = nodeMap.get(id);
      hoverLabel.textContent = ref?.label ?? id;
      hoverLabel.classList.add('visible');
    } else {
      hoverLabel.classList.remove('visible');
    }
  }
  if (id) {
    hoverLabel.style.left = (e.clientX + 12) + 'px';
    hoverLabel.style.top = (e.clientY + 14) + 'px';
  }
});

// Track the node a left-mouse press landed on (separate from the
// drag tracker — capability + preview nodes are clickable but NOT
// draggable, so they need their own candidate tracker).
let clickCandidate = null;

canvas.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  // Ignore clicks while the camera is animating in/out of focus.
  if (cameraState === 'focusing' || cameraState === 'unfocusing') return;
  const hit = pickNodeUnderMouse();
  // Empty-space single click does nothing — falls through to
  // OrbitControls so the user can drag to orbit. To leave focus mode
  // use Esc, double-click empty space, the breadcrumb IRIS chip, or
  // the IRIS CORTEX brand in the titlebar.
  if (!hit) return;
  const ref = nodeMap.get(hit.object.userData.nodeId);
  if (!ref) return;
  clickCandidate = ref;
  pressTime = performance.now();
  // Drag is allowed only for full-size, non-capability nodes that
  // ARE NOT the currently-focused node. The focused node should yield
  // to OrbitControls so click+drag on it rotates the camera around
  // it rather than dragging the node itself across the scene.
  const isFocusedNode = ref.id === focusedNodeId;
  if (!ref.isCapability && !ref.isPreview && !isFocusedNode) {
    draggingRef = ref;
    ref.dragged = true;
    const normal = new THREE.Vector3().subVectors(camera.position, ref.mesh.position).normalize();
    dragPlane.setFromNormalAndCoplanarPoint(normal, ref.mesh.position);
    dragOffset.subVectors(hit.point, ref.mesh.position);
    dragLast.copy(ref.mesh.position);
    controls.enabled = false;
  }
});

canvas.addEventListener('mouseup', (e) => {
  const held = performance.now() - pressTime;
  let moved = false;
  let releasedDrag = false;
  if (draggingRef) {
    moved = draggingRef.mesh.position.distanceTo(dragLast) > 0.01;
    draggingRef.dragged = false;
    draggingRef = null;
    releasedDrag = true;
  }
  const quickClick = clickCandidate && held < 240 && !moved;
  if (quickClick) {
    // Click on the currently-focused node → return to overview.
    if (clickCandidate.id === focusedNodeId) {
      unfocusNode();
    } else {
      focusNode(clickCandidate.id);
      // focusNode disables controls. Don't re-enable here.
    }
  } else if (releasedDrag) {
    // Drag finished without a focus click — restore orbit controls.
    controls.enabled = true;
  }
  clickCandidate = null;
});

canvas.addEventListener('dblclick', () => {
  const hit = pickNodeUnderMouse();
  // Double-click on empty space → return to overview.
  if (!hit) {
    if (focusedNodeId !== null) unfocusNode();
    return;
  }
  if (!pythonBridge) return;
  const ref = nodeMap.get(hit.object.userData.nodeId);
  if (!ref) return;
  // If the node carries a file path (preview file-children do), open it
  // via the OS default handler. Otherwise fall back to the legacy leaf
  // notification so older callers keep working.
  if (ref.path) {
    try { pythonBridge.openFile(ref.path); } catch (_) {}
    // Brief activity flare on the leaf so the user gets visual ack.
    ref.pulseActivity(0.9, 500);
    return;
  }
  if (ref.isLeaf) {
    try { pythonBridge.leafOpened(ref.id); } catch (_) {}
  }
});

// ─── keybindings ───────────────────────────────────────────────────

const debugEl = document.getElementById('debug');
let bloomEnabled = true;
window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') {
    camera.position.set(0, 18, 90);
    controls.target.set(0, 0, 0);
  } else if (e.key === 'd' || e.key === 'D') {
    debugEl.classList.toggle('visible');
  } else if (e.key === 'b' || e.key === 'B') {
    bloomEnabled = !bloomEnabled;
    bloomPass.enabled = bloomEnabled;
  } else if (e.key === 'f' || e.key === 'F') {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  } else if (e.key === 'Escape') {
    if (focusedNodeId !== null) unfocusNode();
  }
});

// Breadcrumb IRIS chip → return to overview.
const _bcHome = document.querySelector('#breadcrumb [data-bc-home]');
if (_bcHome) {
  _bcHome.addEventListener('click', () => unfocusNode());
}

// Fade out the hint after a few seconds.
setTimeout(() => {
  const hint = document.getElementById('hint');
  if (hint) hint.classList.add('fade');
}, 6000);

// ─── animation loop ────────────────────────────────────────────────

let lastFrame = performance.now();
let fpsAccum = 0;
let fpsFrames = 0;
let fpsLast = lastFrame;

// Render-pause gates. Cortex must NOT compete with Touchless's camera
// pipeline for GPU when the user can't even see it. Two independent
// gates ORed together — render only when BOTH say "go":
//   * `hostPaused`       — Qt side (window.cortexSetPaused). Fires on
//                          panel collapse / Iris-window hide / minimize.
//   * `visibilityPaused` — Page Visibility API. Catches OS-level hides
//                          that Qt's collapse path doesn't always emit
//                          (browser tabs, screensaver, etc.).
// When either is true we skip all Three.js work (composer pass,
// particle updates, lerps, OrbitControls.update) and let RAF idle.
// Idle cost is negligible; rendering cost is 15-30% GPU on a mid
// laptop.
let hostPaused = false;
let visibilityPaused = false;
window.cortexSetPaused = function(v) { hostPaused = !!v; };
document.addEventListener('visibilitychange', () => {
  visibilityPaused = !!document.hidden;
});

function frame() {
  // Fast path when paused — never touch the GPU.
  if (hostPaused || visibilityPaused) {
    requestAnimationFrame(frame);
    return;
  }
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastFrame) / 1000); // cap dt for stalls
  lastFrame = now;

  // Camera animation (focus mode fly-in / fly-out). Runs before any
  // other update so the controls.target is correct when OrbitControls
  // resumes at the end of the animation.
  if (cameraState === 'focusing' || cameraState === 'unfocusing') {
    const t = Math.min(1, (now - cameraAnimStart) / CAMERA_ANIM_MS);
    const eased = 1 - Math.pow(1 - t, 3);
    if (cameraStartPos && cameraEndPos) {
      camera.position.lerpVectors(cameraStartPos, cameraEndPos, eased);
    }
    if (cameraStartTarget && cameraEndTarget) {
      controls.target.lerpVectors(cameraStartTarget, cameraEndTarget, eased);
    }
    if (t >= 1) {
      cameraState = cameraState === 'focusing' ? 'focused' : 'overview';
      controls.enabled = true;
    }
  } else if (cameraState === 'focused' && focusedNodeId !== null) {
    // Stay locked to the focused node so drag-orbit always rotates
    // around it cleanly, even if the node nudges slightly.
    const fNode = nodeMap.get(focusedNodeId);
    if (fNode) controls.target.copy(fNode.mesh.position);
  }

  // Refresh the per-orbit-center kid count BEFORE any node updates
  // run, so each NodeRef.update sees the correct count for its
  // auto-fit radius computation.
  recomputeOrbitCounts();

  // Twinkle starfield — each layer oscillates opacity at its own
  // slow phase so the void breathes instead of sitting flat.
  for (const layer of starLayers) {
    const v = 0.85 + 0.15 * Math.sin(now * 0.0007 * layer.twinkleSpeed);
    layer.mat.opacity = layer.baseOpacity * v;
  }
  // Drift the ambient dust through the scene volume.
  updateAmbientParticles(dt);

  // Smooth audio toward target — gives the core an inertia-y reactivity.
  coreAudioDecay += (coreAudio - coreAudioDecay) * Math.min(1, dt * 8);
  // Idle breathing + audio reactivity on core.
  const breath = 1.0 + 0.03 * Math.sin(now * 0.0017) + coreAudioDecay * 0.18;
  coreInner.scale.setScalar(breath);
  // Halo is a Sprite — its scale defines its world size, so multiply
  // by the tracked base size rather than overwriting it.
  coreHalo.scale.setScalar(_coreHaloBaseSize * breath * 1.05);
  // Audio adds a brief boost on top of the resting emissive (set by
  // setCoreState). Modest because the sprite halo carries most of
  // the visible "glow."
  if (coreAudioDecay > 0.02) {
    coreInner.material.emissiveIntensity = 0.7 + coreIntensity * 0.8 + coreAudioDecay * 0.9;
  }

  // Update all nodes.
  for (const ref of nodeMap.values()) {
    ref.update(dt, now);
  }

  // Sweep finished fades.
  for (const id of [...nodeMap.keys()]) {
    const ref = nodeMap.get(id);
    if (ref.fading && now >= ref.fadeUntil) {
      // Drop any edges that touch this node.
      for (let i = edges.length - 1; i >= 0; i--) {
        const e = edges[i];
        if (e.userData.fromId === id || e.userData.toId === id) {
          scene.remove(e);
          e.geometry.dispose();
          e.material.dispose();
          edges.splice(i, 1);
        }
      }
      // Keep childCount bookkeeping consistent so a re-focus
      // doesn't think a parent still has children that left.
      const parentKey = ref.parentId ?? 'core';
      if (childCount.has(parentKey)) {
        const c = childCount.get(parentKey) - 1;
        if (c <= 0) childCount.delete(parentKey);
        else childCount.set(parentKey, c);
      }
      if (childrenByParent.has(parentKey)) {
        childrenByParent.get(parentKey).delete(id);
        if (childrenByParent.get(parentKey).size === 0) {
          childrenByParent.delete(parentKey);
        }
      }
      if (focusedNodeId === id) unfocusNode();
      ref.dispose();
      nodeMap.delete(id);
    }
  }

  updateEdges();
  updateSiblingEdges(now);
  updatePulses(now);

  controls.update();
  composer.render();

  // FPS + debug overlay.
  fpsFrames++;
  if (now - fpsLast > 500) {
    fpsAccum = fpsFrames * 1000 / (now - fpsLast);
    fpsFrames = 0;
    fpsLast = now;
    document.getElementById('fps').textContent = fpsAccum.toFixed(0);
    document.getElementById('nodecount').textContent = nodeMap.size.toString();
  }

  // Event rate (per second, sliding 1s window).
  if (now - evtWindowStart > 1000) {
    evtRate = evtCount;
    evtCount = 0;
    evtWindowStart = now;
    document.getElementById('evrate').textContent = evtRate.toString();
  }

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ─── resize handling ───────────────────────────────────────────────

window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
});
