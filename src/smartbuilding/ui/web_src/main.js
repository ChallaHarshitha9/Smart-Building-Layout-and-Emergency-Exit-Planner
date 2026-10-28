import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';

const H = 3.0, FH = 3.4, $ = (s) => document.querySelector(s);
const METHODS = {
  B1: ['B1 nearest exit', 'Shortest path to the nearest exit. Static, ignores hazards and exit status.'],
  B2: ['B2 threshold', 'SafeStep-style: removes nodes believed hazardous, then shortest path.'],
  B3: ['B3 oracle', 'Dose-aware planner fed ground truth. An upper bound, not deployable.'],
  B4: ['B4 raw obs.', 'Dose-aware planner on the latest raw sensor readings (no filtering).'],
  B5: ['B5 belief', 'Dose-aware planner on the Bayes belief, predicted forward in time.'],
};
const COL = {B1: '#ff7b7b', B2: '#ffb454', B3: '#8b9bb4', B4: '#c58bff', B5: '#4da3ff'};
const MODES = [['inspect', 'Inspect'], ['fire', 'Place fire'], ['block', 'Block exit'], ['second', 'Second fire'], ['detector', 'Fail detector']];
const S = {specs: [], layouts: [], li: 0, L: null, sim: null, fi: 0, playing: false, speed: 1, method: 'B5', compare: '',
  lens: 'truth', mode: 'inspect', events: [], ignition: null, sel: null, routes: true, follow: false, dose: {alert: .5, critical: 3, severe: 8},
  cons: null, day: 100};
const api = async (path, body) => {
  const r = await fetch(path, {method: body ? 'POST' : 'GET', headers: {'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined});
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
};
const toast = (m, ms = 2500) => { const t = $('#toast'); t.textContent = m; t.style.display = 'block'; clearTimeout(toast.h); if (ms) toast.h = setTimeout(() => t.style.display = 'none', ms); };

// ---------------------------------------------------------------- three.js
const canvas = $('#c'), renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, .1, 500);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * .49;
scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x2a3040, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(20, 40, 15); scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshBasicMaterial({color: 0x131a28}));
ground.rotation.x = -Math.PI / 2; ground.position.y = -.12; scene.add(ground);
const grid = new THREE.GridHelper(200, 100, 0x2b3750, 0x1c2538); grid.position.y = -.1; scene.add(grid);
const bGroup = new THREE.Group(), smokeG = new THREE.Group(), fireG = new THREE.Group(), routeG = new THREE.Group(), labelG = new THREE.Group();
scene.add(bGroup, smokeG, fireG, routeG, labelG);
let rooms = {}, exitsM = {}, doorsM = {}, smokeM = [], fireM = {}, people = null, ghost = null, inst = [];
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
const P = (x, y, f = 0, h = 0) => new THREE.Vector3(x - S.ox, f * FH + h, -(y - S.oy));
const PAL = [0x6aa5ff, 0x58c9a0, 0x9aa4c8, 0xb69be0, 0x7fd0e0, 0x8fc0a0, 0x8f9bd6, 0xd0b8e8, 0x6fb5b0, 0xa9b8d8];
const typeIdx = {}; const hue = (t) => new THREE.Color(PAL[(typeIdx[t] ??= Object.keys(typeIdx).length) % PAL.length]);

function label(text, pos, size = 1) {
  const c = document.createElement('canvas'); c.width = 256; c.height = 64; const g = c.getContext('2d');
  g.font = '600 26px system-ui'; g.fillStyle = '#e6ebf5'; g.textAlign = 'center'; g.shadowColor = '#000'; g.shadowBlur = 6; g.fillText(text, 128, 40);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), transparent: true, depthTest: false}));
  sp.scale.set(4 * size, 1 * size, 1); sp.position.copy(pos); sp.renderOrder = 10; return sp;
}
const clear = (g) => { while (g.children.length) { const o = g.children.pop(); o.traverse?.((x) => { x.geometry?.dispose(); x.material?.dispose?.(); }); } };

function buildLayout(L) {
  S.L = L; S.ox = L.size[0] / 2; S.oy = L.size[1] / 2; Object.keys(typeIdx).forEach((k) => delete typeIdx[k]);
  [bGroup, smokeG, fireG, routeG, labelG].forEach(clear); rooms = {}; exitsM = {}; doorsM = {}; smokeM = []; fireM = {};
  for (let f = 0; f < L.floors; f++) {
    const plate = new THREE.Mesh(new THREE.BoxGeometry(L.size[0], .06, L.size[1]), new THREE.MeshStandardMaterial({color: 0x1b2438}));
    plate.position.copy(P(L.size[0] / 2, L.size[1] / 2, f, -.1)); bGroup.add(plate);
    const o = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(L.size[0], .06, L.size[1])), new THREE.LineBasicMaterial({color: 0x4a5a7c})); o.position.copy(plate.position); bGroup.add(o);
  }
  for (const c of L.corridors) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(c.w, .12, c.d), new THREE.MeshStandardMaterial({color: 0x39445e}));
    m.position.copy(P(c.x + c.w / 2, c.y + c.d / 2, c.floor, 0)); bGroup.add(m);
  }
  for (const r of L.rooms) {
    const g = new THREE.Group(), col = hue(r.type);
    const slab = new THREE.Mesh(new THREE.BoxGeometry(r.w, .18, r.d), new THREE.MeshStandardMaterial({color: col}));
    const wg = new THREE.BoxGeometry(r.w, H, r.d);
    const walls = new THREE.Mesh(wg, new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: .1, depthWrite: false}));
    walls.position.y = H / 2;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(wg), new THREE.LineBasicMaterial({color: 0x7f8db0})); edges.position.y = H / 2;
    g.add(slab, walls, edges); g.position.copy(P(r.x + r.w / 2, r.y + r.d / 2, r.floor, 0)); g.userData = {roomId: r.id};
    bGroup.add(g); rooms[r.id] = {g, slab, walls, edges, r, col};
    const lab = label(`${r.label}`, P(r.x + r.w / 2, r.y + r.d / 2, r.floor, H + .6), .9); labelG.add(lab); rooms[r.id].label = lab;
  }
  for (const d of L.doors) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(d.horizontal ? d.width : .35, 2.2, d.horizontal ? .35 : d.width), new THREE.MeshBasicMaterial({color: 0x4da3ff, transparent: true, opacity: .55}));
    m.position.copy(P(d.x, d.y, d.floor, 1.1)); bGroup.add(m); doorsM[d.id] = m;
  }
  for (const e of L.exits) {
    const horiz = /^[ns]/i.test(e.side);
    const m = new THREE.Mesh(new THREE.BoxGeometry(horiz ? e.width : .6, 2.6, horiz ? .6 : e.width), new THREE.MeshBasicMaterial({color: 0x3ecf8e, transparent: true, opacity: .75}));
    m.position.copy(P(e.x, e.y, e.floor, 1.3)); m.userData = {exitId: e.id}; bGroup.add(m); exitsM[e.id] = m;
    labelG.add(label(e.id, P(e.x, e.y, e.floor, 3.2), .8));
  }
  L.nodes.forEach((n, i) => {
    const c = L.cells[n.id]; if (!c || c.w === undefined) { smokeM.push(null); return; }
    const m = new THREE.Mesh(new THREE.BoxGeometry(Math.max(.3, c.w - .1), H - .3, Math.max(.3, c.d - .1)), new THREE.MeshBasicMaterial({color: 0x333333, transparent: true, opacity: 0, depthWrite: false}));
    m.position.copy(P(c.x + c.w / 2, c.y + c.d / 2, c.floor, H / 2)); m.visible = false; smokeG.add(m); smokeM.push(m);
    const f = new THREE.Group();
    for (let k = 0; k < 3; k++) {
      const cone = new THREE.Mesh(new THREE.ConeGeometry(.55 - k * .1, 1.7 - k * .3, 10), new THREE.MeshBasicMaterial({color: [0xff4d1a, 0xff9f1a, 0xffe066][k], transparent: true, opacity: .85}));
      cone.position.y = .85 - k * .1; f.add(cone);
    }
    f.position.copy(P(c.x + c.w / 2, c.y + c.d / 2, c.floor, 0)); f.visible = false; fireG.add(f); fireM[i] = f;
  });
  fitCamera('iso'); applyReveal();
}

function fitCamera(kind) {
  const W = S.L.size[0], D = S.L.size[1], R = Math.max(W, D), fl = S.L.floors;
  const t = new THREE.Vector3(0, (fl - 1) * FH / 2 + 1, 0); controls.target.copy(t);
  const pos = {iso: [R * .45, R * .95, R * 1.15], top: [0, R * 1.5, .05], side: [0, R * .3, R * 1.5]}[kind] || [R * .45, R * .95, R * 1.15];
  camera.position.set(t.x + pos[0], t.y + pos[1], t.z + pos[2]); controls.update();
}

// ---------------------------------------------------------------- people
const doseCol = (d) => { const z = S.dose; return d >= z.severe ? [1, .3, .3] : d >= z.critical ? [1, .62, .26] : d >= z.alert ? [1, .82, .4] : [.24, .81, .56]; };
function buildPeople() {
  for (const o of [people, ghost]) if (o) { scene.remove(o); o.dispose?.(); }
  inst = []; people = ghost = null; if (!S.sim) return;
  S.sim.groups.forEach((g, gi) => { const n = Math.min(g.size, 14); for (let k = 0; k < n; k++) { const a = k * 2.399, r = .28 * Math.sqrt(k); inst.push([gi, Math.cos(a) * r, Math.sin(a) * r]); } });
  people = new THREE.InstancedMesh(new THREE.CapsuleGeometry(.2, .8, 4, 8), new THREE.MeshStandardMaterial({color: 0xffffff}), inst.length);
  ghost = new THREE.InstancedMesh(new THREE.TorusGeometry(.28, .05, 6, 14), new THREE.MeshBasicMaterial({color: 0xb8c4dc, transparent: true, opacity: .7}), inst.length);
  ghost.rotation.x = Math.PI / 2; people.frustumCulled = ghost.frustumCulled = false; scene.add(people, ghost);
}
const mat = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), pv = new THREE.Vector3(), col = new THREE.Color();
function placePeople(mesh, method, f0, f1, a, withColor) {
  const fr0 = S.sim.methods[method].frames, A = fr0[f0], B = fr0[f1];
  if (!A) { mesh.visible = false; return; } mesh.visible = true;
  inst.forEach(([gi, ox, oy], i) => {
    const ga = A.g[gi], gb = B.g[gi]; const gone = ga[4] && gb[4];
    const x = ga[0] + (gb[0] - ga[0]) * a + ox, y = ga[1] + (gb[1] - ga[1]) * a + oy;
    pv.copy(P(x, y, ga[2], mesh === people ? .65 : .08)); sc.setScalar(gone ? 0 : 1);
    mat.compose(pv, q, sc); mesh.setMatrixAt(i, mat);
    if (withColor) { const c = doseCol(ga[3] + (gb[3] - ga[3]) * a); mesh.setColorAt(i, col.setRGB(c[0], c[1], c[2])); }
  });
  mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------------------------------------------------------------- frame update
function render() {
  if (!S.sim) return;
  const M = S.sim.methods[S.method], fr = M.frames, i0 = Math.floor(S.fi), i1 = Math.min(i0 + 1, fr.length - 1), a = S.fi - i0, F = fr[i0];
  placePeople(people, S.method, i0, i1, a, true);
  if (S.compare && S.sim.methods[S.compare]) placePeople(ghost, S.compare, Math.min(i0, S.sim.methods[S.compare].frames.length - 1), Math.min(i1, S.sim.methods[S.compare].frames.length - 1), a, false); else ghost.visible = false;
  const belief = S.lens === 'belief' && F.b;
  smokeM.forEach((m, i) => {
    if (!m) return; const v = belief ? F.b[i] / 100 : F.s[i] / 100;
    m.visible = v > .02; m.material.opacity = Math.min(.8, v * (belief ? .75 : .95));
    m.material.color.set(belief ? new THREE.Color().setHSL(.62 - .62 * v, .8, .5) : 0x2b2b2b);
  });
  const fset = new Set(F.f); for (const k in fireM) fireM[k].visible = fset.has(+k);
  for (const [id, m] of Object.entries(exitsM)) { const blocked = F.x.includes(S.L.nodes.findIndex((n) => n.region === id)); m.material.color.set(blocked ? 0xff4d4d : 0x3ecf8e); }
  drawRoutes(i0);
  const done = F.g.reduce((s, g, gi) => s + (g[4] ? S.sim.groups[gi].size : 0), 0), tot = S.sim.groups.reduce((s, g) => s + g.size, 0);
  $('#tlabel').textContent = `t = ${(F.t).toFixed(0)} s   evacuated ${done}/${tot}   [${S.method}]`;
  $('#scrub').value = S.fi; drawCharts();
  if (S.follow) { const live = F.g.filter((g) => !g[4]); if (live.length) { const cx = live.reduce((s, g) => s + g[0], 0) / live.length, cy = live.reduce((s, g) => s + g[1], 0) / live.length; controls.target.lerp(P(cx, cy, live[0][2], 1), .08); } }
}
function drawRoutes(i) {
  clear(routeG); if (!S.routes) return; const fr = S.sim.methods[S.method].frames; let r = null;
  for (let k = i; k >= 0; k--) if (fr[k].r) { r = fr[k].r; break; } if (!r) return;
  S.sim.groups.forEach((g, gi) => { const path = r[g.id]; if (!path || path.length < 2 || fr[i].g[gi][4]) return;
    const pts = path.map((ni) => { const n = S.L.nodes[ni]; return P(n.x, n.y, n.floor, .25); });
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: 0xcfe0ff, transparent: true, opacity: .55})); routeG.add(l); });
}

// ---------------------------------------------------------------- charts & cards
function series(M) {
  const g = S.sim.groups, tot = g.reduce((s, x) => s + x.size, 0);
  return M.frames.map((F) => ({t: F.t, dose: F.g.reduce((s, x, i) => s + x[3] * g[i].size, 0) / tot, ev: F.g.reduce((s, x, i) => s + (x[4] ? g[i].size : 0), 0)}));
}
function drawChart(cv, key, tmax, ymax) {
  const g = cv.getContext('2d'), w = cv.width, h = cv.height; g.clearRect(0, 0, w, h); g.strokeStyle = '#2b3750'; g.strokeRect(30.5, 4.5, w - 36, h - 18);
  g.fillStyle = '#93a0b8'; g.font = '10px system-ui'; g.fillText(ymax.toFixed(ymax < 10 ? 1 : 0), 2, 12); g.fillText('0', 18, h - 8); g.fillText(tmax.toFixed(0) + ' s', w - 34, h - 3);
  for (const [m, ser] of Object.entries(S.series)) { g.beginPath(); g.strokeStyle = COL[m]; g.lineWidth = m === S.method ? 2.4 : 1.2;
    ser.forEach((p, i) => { const x = 30 + p.t / tmax * (w - 36), y = h - 14 - (p[key] / ymax) * (h - 18); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    const last = ser[ser.length - 1]; g.lineTo(w - 6, h - 14 - (last[key] / ymax) * (h - 18)); g.stroke(); }
  const F = S.sim.methods[S.method].frames[Math.floor(S.fi)]; const x = 30 + F.t / tmax * (w - 36); g.strokeStyle = '#fff6'; g.beginPath(); g.moveTo(x, 4); g.lineTo(x, h - 14); g.stroke();
}
function drawCharts() {
  if (!S.series) return; const tmax = Math.max(...Object.values(S.series).map((s) => s[s.length - 1].t), 10);
  const dm = Math.max(.5, ...Object.values(S.series).map((s) => Math.max(...s.map((p) => p.dose)))), em = S.sim.groups.reduce((s, x) => s + x.size, 0);
  drawChart($('#ch1'), 'dose', tmax, dm * 1.05); drawChart($('#ch2'), 'ev', tmax, em);
}
function buildCards() {
  const box = $('#cards'); box.innerHTML = ''; const ms = Object.keys(S.sim.methods), best = ms.reduce((a, b) => S.sim.methods[a].summary.mean_dose <= S.sim.methods[b].summary.mean_dose ? a : b);
  for (const m of ms) { const s = S.sim.methods[m].summary, d = document.createElement('div'); d.className = 'card' + (m === S.method ? ' on' : '') + (m === best ? ' best' : '');
    d.innerHTML = `<h4><span style="color:${COL[m]}">${METHODS[m][0]}</span><span class="note">${m === best ? 'lowest dose' : ''}</span></h4><div class="kv"><span><b>${(s.completion * 100).toFixed(0)}%</b>evacuated</span><span><b>${s.mean_evac_s == null ? '–' : s.mean_evac_s.toFixed(0) + ' s'}</b>mean time</span><span><b>${s.mean_dose.toFixed(2)}</b>dose proxy</span><span><b>${s.dose_levels.severe}</b>severe</span><span><b>${s.peak_queue_delay_s.toFixed(0)} s</b>peak queue</span><span><b>${s.invalid_decisions}</b>invalid</span></div>`;
    d.onclick = () => { S.method = m; $('#mdesc').textContent = METHODS[m][1]; buildCards(); S.fi = Math.min(S.fi, S.sim.methods[m].frames.length - 1); setScrub(); render(); }; box.appendChild(d); }
  $('#mdesc').textContent = METHODS[S.method][1];
  $('#cmp').innerHTML = '<option value="">none</option>' + ms.filter((m) => m !== S.method).map((m) => `<option value="${m}">${METHODS[m][0]}</option>`).join('');
  $('#cmp').value = ms.includes(S.compare) && S.compare !== S.method ? S.compare : ''; S.compare = $('#cmp').value;
}
const setScrub = () => { const n = S.sim.methods[S.method].frames.length - 1; $('#scrub').max = n; };

// ---------------------------------------------------------------- UI: layout panel
function showLayout(i) {
  S.li = i; const L = S.layouts[i]; buildLayout(L); S.sim = null; S.series = null; buildPeople(); $('#cards').innerHTML = ''; $('#tlabel').textContent = '';
  $('#cands').innerHTML = S.layouts.map((l, k) => `<span class="chip ${k === i ? 'on' : ''}" data-i="${k}">Layout ${k + 1} · ${(l.csr * 100).toFixed(0)}%</span>`).join('');
  $('#cands').querySelectorAll('.chip').forEach((c) => c.onclick = () => showLayout(+c.dataset.i));
  $('#csr').textContent = `${L.hard.filter((h) => h.ok).length}/${L.hard.length} met`;
  $('#rules').innerHTML = L.hard.map((h, k) => `<div class="rule" data-k="${k}"><i class="dot ${h.ok ? 'ok' : 'no'}"></i><div>${h.id}<small>${h.detail || ''}</small></div></div>`).join('');
  $('#rules').querySelectorAll('.rule').forEach((r) => r.onclick = () => flash(L.hard[+r.dataset.k].entities));
  $('#soft').innerHTML = L.soft.map((s) => `<div style="margin:5px 0"><div class="row" style="margin:0;justify-content:space-between"><span>${s.name}</span><span class="note">${s.cost.toFixed(2)}</span></div><div class="bar"><i style="width:${Math.min(100, s.cost * 100)}%"></i></div></div>`).join('');
  S.events = []; S.ignition = null; renderEvents(); $('#runinfo').textContent = 'Choose Place fire and click a room.'; if (S.cons) loadConstruction();
}
function flash(ids) {
  const hit = (id) => rooms[id] || doorsM[id] || exitsM[id]; const objs = ids.map(hit).filter(Boolean);
  objs.forEach((o) => { if (o.edges) { o.edges.material.color.set(0xff6b6b); o.walls.material.opacity = .35; } else o.material.color.set(0xff6b6b); });
  setTimeout(() => { Object.values(rooms).forEach((r) => { r.edges.material.color.set(0x7f8db0); r.walls.material.opacity = S.sel === r.r.id ? .3 : .1; }); Object.values(doorsM).forEach((d) => d.material.color.set(0x4da3ff)); Object.values(exitsM).forEach((d) => d.material.color.set(0x3ecf8e)); }, 2200);
}
function select(id) {
  S.sel = id; Object.values(rooms).forEach((r) => { r.walls.material.opacity = r.r.id === id ? .3 : .1; r.edges.material.color.set(r.r.id === id ? 0xffb454 : 0x7f8db0); });
  const r = rooms[id]?.r; if (!r) { $('#sel').textContent = 'Click a room.'; return; }
  let extra = ''; if (S.sim) { const F = S.sim.methods[S.method].frames[Math.floor(S.fi)], ni = S.L.nodes.findIndex((n) => n.region === id); extra = `<br>smoke (truth) ${F.s[ni]}%${F.b ? ` · belief hazard ${F.b[ni]}%` : ''}`; }
  $('#sel').innerHTML = `<b style="color:#e6ebf5">${r.label}</b> (${r.id})<br>area ${r.area} m² · load ${r.load} · floor ${r.floor}<br>window ${r.needs_window ? 'required' : 'not required'}${extra}`;
}

// ---------------------------------------------------------------- UI: scenario
const MODELAB = {fire: 'Ignition', second: 'Second fire', detector: 'Detector failure', block: 'Exit blocked'};
function renderEvents() {
  $('#evlist').innerHTML = (S.ignition ? [`<span class="chip on">🔥 ${S.ignition}</span>`] : []).concat(S.events.map((e, k) => `<span class="chip" data-k="${k}">${MODELAB[e.mode]} ${e.target}${e.type === 'detector_fail' ? '' : ' @' + e.time + 's'} ✕</span>`)).join('');
  $('#evlist').querySelectorAll('[data-k]').forEach((c) => c.onclick = () => { S.events.splice(+c.dataset.k, 1); renderEvents(); });
}
function pick(ev) {
  const b = canvas.getBoundingClientRect(); mouse.set((ev.clientX - b.left) / b.width * 2 - 1, -((ev.clientY - b.top) / b.height) * 2 + 1); ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(bGroup.children, true);
  if (S.mode === 'block') for (const h of hits) { let o = h.object; while (o && !o.userData.exitId) o = o.parent; if (o) return o.userData; }
  for (const h of hits) { let o = h.object; while (o && !o.userData.roomId && !o.userData.exitId) o = o.parent; if (o) return o.userData; } return null;
}
let down = null; canvas.addEventListener('pointerdown', (e) => down = [e.clientX, e.clientY]);
canvas.addEventListener('pointerup', async (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return; const hit = pick(e); if (!hit) return; const t = +$('#btime').value || 0;
  if (hit.roomId) select(hit.roomId);
  if (S.mode === 'fire' && hit.roomId) { S.ignition = hit.roomId; renderEvents(); await run(); }
  else if (S.mode === 'second' && hit.roomId) { S.events.push({mode: 'second', type: 'second_fire', target: hit.roomId, time: t}); renderEvents(); if (S.ignition) await run(); }
  else if (S.mode === 'detector' && hit.roomId) { S.events.push({mode: 'detector', type: 'detector_fail', target: hit.roomId, time: 0}); renderEvents(); if (S.ignition) await run(); }
  else if (S.mode === 'block' && hit.exitId) { S.events.push({mode: 'block', type: 'exit_blocked', target: hit.exitId, time: t}); renderEvents(); if (S.ignition) await run(); }
});
async function run() {
  if (!S.ignition) { toast('Place a fire first: choose Place fire and click a room'); return; }
  toast('Simulating all methods…', 0); $('#run').disabled = true;
  try {
    const res = await api('/api/simulate', {layout: S.L.key, ignition: S.ignition, events: S.events.map(({type, target, time}) => ({type, target, time})), period: $('#period').value, hazard: $('#hazard').value, seed: +$('#sseed').value, horizon: 300});
    S.sim = res; S.series = {}; for (const m in res.methods) S.series[m] = series(res.methods[m]); S.fi = 0; if (!res.methods[S.method]) S.method = Object.keys(res.methods)[0];
    buildPeople(); buildCards(); setScrub(); render(); $('#runinfo').textContent = `${res.groups.reduce((s, g) => s + g.size, 0)} people in ${res.groups.length} groups · seed ${res.seed}`; S.playing = true; $('#play').textContent = 'Pause'; toast('Done', 800);
  } catch (e) { toast('Error: ' + e.message, 6000); } finally { $('#run').disabled = false; }
}

// ---------------------------------------------------------------- construction & E0
async function loadConstruction() {
  const c = await api('/api/construction', {layout: S.L.key}); S.cons = c; const T = c.total_days || 1;
  $('#gantt').innerHTML = c.tasks.map((t) => `<div class="grow"><span class="glabel" title="${t.id}">${t.type} ${t.target || ''}</span><span class="gtrack"><i class="gbar ${t.critical ? 'crit' : ''}" style="left:${t.start / T * 100}%;width:${Math.max(.8, (t.end - t.start) / T * 100)}%"></i></span></div>`).join('');
  $('#cinfo').textContent = `${c.tasks.length} tasks · ${c.total_days.toFixed(0)} days · cost ${c.total_cost.toFixed(0)} units · critical path in amber`; applyReveal();
}
function applyReveal() {
  if (!S.L) return; const c = S.cons, day = S.day / 100 * (c?.total_days || 1), building = !$('#p-build').classList.contains('hidden');
  for (const [id, r] of Object.entries(rooms)) { let f = 1;
    if (building && c) { const ts = c.tasks.filter((t) => t.target === id); if (ts.length) { let d = 0, w = 0; ts.forEach((t) => { const l = Math.max(.01, t.end - t.start); w += l; d += l * Math.min(1, Math.max(0, (day - t.start) / l)); }); f = d / w; } }
    r.g.scale.y = Math.max(.02, f); r.slab.scale.y = 1; r.g.visible = f > .01; if (r.label) r.label.visible = f > .3; }
  const done = (id) => { if (!(building && c)) return true; const ts = c.tasks.filter((t) => t.target === id); return !ts.length || ts.every((t) => t.end <= day); };
  for (const [id, m] of Object.entries(doorsM)) m.visible = done(id);
  for (const [id, m] of Object.entries(exitsM)) m.visible = done(id);
  $('#daylab').textContent = building && c ? `day ${day.toFixed(0)}` : '';
}
$('#day').oninput = (e) => { S.day = +e.target.value; applyReveal(); };
$('#rune0').onclick = async () => {
  $('#e0out').textContent = 'Running…'; const r = await api('/api/e0');
  $('#e0out').innerHTML = `<table><tr><th>method / response</th><th>mean dose</th><th>exits used</th></tr>${Object.keys(r.mean_dose).map((k) => `<tr><td>${k}</td><td>${r.mean_dose[k].toFixed(2)}</td><td>${Object.entries(r.exit_counts[k]).map(([a, b]) => a + '×' + b).join(' ')}</td></tr>`).join('')}</table><p class="note">${r.check_passed ? '✓' : '✗'} ${r.note}</p>`;
};

// ---------------------------------------------------------------- wiring
document.querySelectorAll('.tab').forEach((t) => t.onclick = () => { document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
  for (const k of ['evac', 'build', 'e0']) $('#p-' + k).classList.toggle('hidden', k !== t.dataset.tab); $('#timeline').style.visibility = t.dataset.tab === 'evac' ? 'visible' : 'hidden';
  if (t.dataset.tab === 'build' && !S.cons && S.L) loadConstruction(); applyReveal(); });
$('#modes').innerHTML = MODES.map(([k, v]) => `<span class="chip ${k === 'inspect' ? 'on' : ''}" data-m="${k}">${v}</span>`).join('');
$('#modes').querySelectorAll('.chip').forEach((c) => c.onclick = () => { S.mode = c.dataset.m; $('#modes').querySelectorAll('.chip').forEach((x) => x.classList.toggle('on', x === c)); canvas.style.cursor = S.mode === 'inspect' ? 'default' : 'crosshair'; });
$('#run').onclick = run; $('#cmp').onchange = (e) => { S.compare = e.target.value; render(); };
$('#play').onclick = () => { S.playing = !S.playing; $('#play').textContent = S.playing ? 'Pause' : 'Play'; };
$('#restart').onclick = () => { S.fi = 0; S.playing = true; $('#play').textContent = 'Pause'; };
$('#scrub').oninput = (e) => { S.fi = +e.target.value; S.playing = false; $('#play').textContent = 'Play'; render(); };
$('#speed').onchange = (e) => S.speed = +e.target.value;
$('#routes').onclick = (e) => { S.routes = !S.routes; e.target.classList.toggle('on', S.routes); if (S.sim) render(); };
$('#lensb').onclick = (e) => { S.lens = S.lens === 'truth' ? 'belief' : 'truth'; e.target.textContent = 'Lens: ' + S.lens; if (S.sim) render(); };
document.querySelectorAll('[data-cam]').forEach((b) => b.onclick = () => { if (b.dataset.cam === 'follow') { S.follow = !S.follow; b.classList.toggle('on', S.follow); } else { S.follow = false; $('#follow').classList.remove('on'); fitCamera(b.dataset.cam); } });
$('#solve').onclick = solveNow;
async function solveNow() {
  toast('Solving…', 0); try { const f = $('#spec').value; const r = await api('/api/solve', {spec: f, seed: +$('#seed').value, max_layouts: +$('#k').value});
    S.dose = r.dose_levels; S.layouts = r.layouts; $('#solveinfo').textContent = `${r.info.feasible} feasible, ${r.info.infeasible} infeasible · ${r.info.time_s}s${r.info.conflict ? ' · ' + r.info.conflict : ''}`;
    if (!r.layouts.length) { toast('No layout found'); return; } showLayout(0); toast('Layouts ready', 900);
  } catch (e) { toast('Error: ' + e.message, 6000); } }
function resize() { const b = $('#stage').getBoundingClientRect(); renderer.setSize(b.width, b.height, false); camera.aspect = b.width / b.height; camera.updateProjectionMatrix(); }
addEventListener('resize', resize);
let last = performance.now();
function loop(now) {
  const dt = Math.min(.1, (now - last) / 1000); last = now; controls.update();
  if (S.sim && S.playing) { const n = S.sim.methods[S.method].frames.length - 1; S.fi = Math.min(n, S.fi + dt * S.speed / S.sim.tick_s * 2); if (S.fi >= n) { S.playing = false; $('#play').textContent = 'Play'; } render(); }
  const fl = 1 + .15 * Math.sin(now / 90); for (const k in fireM) if (fireM[k].visible) { fireM[k].scale.set(fl, 1 + .25 * Math.sin(now / 70 + +k), fl); }
  renderer.render(scene, camera); requestAnimationFrame(loop);
}
(async () => {
  resize(); S.specs = await api('/api/specs'); $('#spec').innerHTML = S.specs.map((s) => `<option value="${s.file}">${s.label}</option>`).join('');
  $('#period').innerHTML = (S.specs[0]?.periods || ['typical']).map((p) => `<option>${p}</option>`).join('');
  $('#spec').onchange = () => { const s = S.specs.find((x) => x.file === $('#spec').value); $('#period').innerHTML = (s.periods || ['typical']).map((p) => `<option>${p}</option>`).join(''); solveNow(); };
  await solveNow(); requestAnimationFrame(loop); window.__app = {S, run, select, showLayout, render, fitCamera, screen: (id) => { const r = rooms[id]?.r; const v = P(r.x + r.w / 2, r.y + r.d / 2, r.floor, 1).project(camera); const b = canvas.getBoundingClientRect(); return [b.left + (v.x + 1) / 2 * b.width, b.top + (1 - v.y) / 2 * b.height]; }, exitScreen: (id) => { const m = exitsM[id]; const v = m.position.clone().project(camera); const b = canvas.getBoundingClientRect(); return [b.left + (v.x + 1) / 2 * b.width, b.top + (1 - v.y) / 2 * b.height]; }}; window.__ready = true;
})();
