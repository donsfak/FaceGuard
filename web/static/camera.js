// camera.js — scanner temps réel.
// Le navigateur capture la webcam, envoie une image JPEG au serveur par WebSocket,
// reçoit la liste des visages (cadre, identité, score, vivacité) et les dessine.
(() => {
  'use strict';
  const { esc, avatar, fmtTime, isToday, toast, UNKNOWN, SPOOF } = window.FG;

  // --- Éléments -----------------------------------------------------------
  const $ = id => document.getElementById(id);
  const stage = $('stage'), stageCard = $('stage-card');
  const video = $('videoElement'), canvas = $('overlayCanvas'), ctx = canvas.getContext('2d');
  const placeholder = $('stage-placeholder'), placeholderText = $('placeholder-text'), retryBtn = $('camera-retry');
  const livePill = $('live-pill'), liveText = $('live-text'), metrics = $('stage-metrics');
  const faceList = $('face-list'), faceEmpty = $('face-empty'), faceCount = $('face-count');
  const eventList = $('event-list'), eventEmpty = $('event-empty');
  const banner = $('welcome-banner');
  const btnPause = $('btn-pause'), btnMirror = $('btn-mirror'), btnKps = $('btn-kps'), btnFull = $('btn-fullscreen');

  const capture = document.createElement('canvas');
  const captureCtx = capture.getContext('2d');

  // Couleurs (un <canvas> ne lit pas les variables CSS)
  const C = { ok: '#2DD4BF', pending: '#60A5FA', unknown: '#F59E0B', danger: '#EF4444', kp: '#5EEAD4', ink: '#042F2A' };

  // --- Préférences mémorisées ---------------------------------------------
  const pref = (k, d) => { try { const v = localStorage.getItem('fg.' + k); return v === null ? d : v === '1'; } catch { return d; } };
  const savePref = (k, v) => { try { localStorage.setItem('fg.' + k, v ? '1' : '0'); } catch { /* navigation privée */ } };

  const state = {
    stream: null, ws: null, wsOpen: false, waiting: false, sendTimer: null,
    reconnectAttempts: 0, reconnectTimer: null,
    paused: false, mirrored: pref('mirror', true), showKps: pref('kps', true),
    lastFaces: [], respTimes: [],
  };

  // --- Statut ---------------------------------------------------------------
  function setLive(kind, text) {
    livePill.className = `live-pill ${kind}`;
    liveText.textContent = text;
  }

  // --- Caméra ---------------------------------------------------------------
  function cameraError(err) {
    switch (err && err.name) {
      case 'NotAllowedError': case 'SecurityError':
        return "Accès à la caméra refusé. Autorisez-la via l'icône à gauche de l'adresse du site, puis réessayez.";
      case 'NotFoundError': case 'OverconstrainedError':
        return 'Aucune caméra détectée. Branchez une webcam puis réessayez.';
      case 'NotReadableError': case 'AbortError':
        return 'La caméra est déjà utilisée (autre onglet, page Enrôler, visio…). Fermez-la puis réessayez.';
      default:
        return window.isSecureContext ? `Caméra indisponible (${err && err.message})`
          : 'La caméra exige HTTPS (ou localhost). Ouvrez le site en https://';
    }
  }

  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }

  async function startCamera() {
    stopCamera();
    placeholder.hidden = false; retryBtn.hidden = true;
    placeholderText.textContent = 'Activation de la caméra…';
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } });
      video.srcObject = state.stream;
      await video.play();
      syncSize();
      placeholder.hidden = true;
      scheduleSend(0);
    } catch (err) {
      console.error('Caméra :', err);
      placeholderText.textContent = cameraError(err);
      retryBtn.hidden = false;
      setLive('down', 'Caméra indisponible');
    }
  }
  retryBtn.addEventListener('click', startCamera);

  // Canvas = résolution réelle de la vidéo ; conteneur = même ratio -> cadres alignés.
  function syncSize() {
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    if (!document.fullscreenElement) stage.style.aspectRatio = `${w} / ${h}`;
  }
  video.addEventListener('loadedmetadata', syncSize);

  // --- WebSocket ------------------------------------------------------------
  function connect() {
    clearTimeout(state.reconnectTimer);
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
    state.ws = ws;
    setLive('warn', 'Connexion à l\'IA…');

    ws.onopen = () => {
      state.wsOpen = true; state.reconnectAttempts = 0;
      if (!state.paused) setLive('ok', 'Analyse en direct');
      scheduleSend(0);
    };
    ws.onclose = () => {
      state.wsOpen = false; state.waiting = false;
      state.reconnectAttempts++;
      const delay = Math.min(1000 * state.reconnectAttempts, 5000);
      setLive('down', `IA déconnectée — nouvelle tentative dans ${Math.round(delay / 1000)} s`);
      state.reconnectTimer = setTimeout(connect, delay);
    };
    ws.onerror = () => { /* onclose gère la reconnexion */ };
    ws.onmessage = ev => {
      try {
        const data = JSON.parse(ev.data);
        const now = performance.now();
        state.respTimes.push(now);
        state.respTimes = state.respTimes.filter(t => now - t < 3000);
        if (!state.paused) handleResult(data);
      } catch (e) {
        console.error('Réponse illisible', e);
      } finally {
        state.waiting = false;
        scheduleSend(60);
      }
    };
  }

  function scheduleSend(delay) {
    clearTimeout(state.sendTimer);
    state.sendTimer = setTimeout(sendFrame, delay);
  }

  function sendFrame() {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN || state.waiting || state.paused || document.hidden) return;
    if (!video.videoWidth || video.readyState < 2) { scheduleSend(200); return; }
    // Image envoyée en 640 px de large max : suffisant pour la détection, réseau léger.
    const scale = Math.min(1, 640 / video.videoWidth);
    capture.width = Math.round(video.videoWidth * scale);
    capture.height = Math.round(video.videoHeight * scale);
    captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
    state.scale = scale;
    state.waiting = true;
    state.ws.send(capture.toDataURL('image/jpeg', 0.75));
  }

  // --- Résultats ------------------------------------------------------------
  function handleResult(data) {
    const faces = (data.faces || []).map(f => scaleFace(f, 1 / (state.scale || 1)));
    state.lastFaces = faces;
    draw(faces);
    renderFaces(faces);
    faces.filter(f => f.logged).forEach(addEvent);

    const fps = state.respTimes.length / 3;
    const known = faces.filter(f => f.identity !== UNKNOWN && f.identity !== SPOOF).length;
    metrics.textContent = `${faces.length} visage${faces.length > 1 ? 's' : ''} · ${known} reconnu${known > 1 ? 's' : ''}`
      + (data.processing_ms != null ? ` · ${Math.round(data.processing_ms)} ms · ${fps.toFixed(1)} img/s` : '');
  }

  function scaleFace(f, k) {
    return { ...f, box: f.box.map(v => v * k), kps: (f.kps || []).map(([x, y]) => [x * k, y * k]) };
  }

  function colorFor(f) {
    if (f.identity === SPOOF) return C.danger;
    if (f.identity === UNKNOWN) return C.unknown;
    return f.is_real ? C.ok : C.pending;
  }

  function label(f) {
    if (f.identity === SPOOF) return 'FRAUDE';
    return `${f.identity}  ${Math.round(f.similarity * 100)}%`;
  }

  function draw(faces) {
    syncSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W = canvas.width;
    // Taille du texte adaptée à l'affichage réel (lisible aussi sur téléphone)
    const k = canvas.width / Math.max(canvas.clientWidth, 1);
    const fontPx = Math.round(15 * k), pad = Math.round(6 * k), lh = Math.round(24 * k);

    faces.forEach(f => {
      let [x1, y1, x2, y2] = f.box;
      if (state.mirrored) [x1, x2] = [W - x2, W - x1];
      const color = colorFor(f);

      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, 3 * k);
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 6 * k) : ctx.rect(x1, y1, x2 - x1, y2 - y1);
      ctx.stroke();

      // Nom AU-DESSUS du visage (exigence du sujet)
      ctx.font = `600 ${fontPx}px Inter, system-ui, sans-serif`;
      const text = label(f);
      const tw = ctx.measureText(text).width + pad * 2;
      let lx = Math.max(0, Math.min(x1, W - tw));
      let ly = y1 - lh - 4 * k;
      if (ly < 0) ly = y2 + 4 * k;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(lx, ly, tw, lh, 5 * k) : ctx.rect(lx, ly, tw, lh);
      ctx.fill();
      ctx.fillStyle = color === C.danger ? '#fff' : C.ink;
      ctx.textBaseline = 'middle';
      ctx.fillText(text, lx + pad, ly + lh / 2 + 1);

      // Barre de progression de la vérification de vivacité, sous le cadre
      if (f.identity !== SPOOF && !f.is_real && f.liveness_progress != null) {
        const bw = x2 - x1, by = y2 + 6 * k;
        ctx.fillStyle = 'rgba(0,0,0,.5)'; ctx.fillRect(x1, by, bw, 4 * k);
        ctx.fillStyle = C.pending; ctx.fillRect(x1, by, bw * f.liveness_progress, 4 * k);
      }

      if (state.showKps) {
        ctx.fillStyle = C.kp;
        (f.kps || []).forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(state.mirrored ? W - x : x, y, 3 * k, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });
  }

  // Liste "Devant la caméra" : mise à jour par piste (pas de clignotement des photos)
  function displayName(f) {
    if (f.identity === UNKNOWN) return 'Personne inconnue';
    if (f.identity === SPOOF) return 'Fraude suspectée';
    return f.identity;
  }
  function metaHtml(f) {
    const score = `${Math.round(f.similarity * 100)}%`;
    if (f.identity === SPOOF) return `<span class="badge badge-danger">Photo ou écran détecté</span>`;
    if (f.identity === UNKNOWN) return `<span>Non enregistré(e) · meilleure ressemblance ${score}</span>`;
    if (f.is_real) return `<span class="badge badge-ok">Vérifié</span><span>confiance ${score}</span>`;
    const p = Math.round((f.liveness_progress || 0) * 100);
    return `<span>Vérification ${p}%</span></div><div class="progress-mini"><span style="width:${p}%"></span>`;
  }
  function renderFaces(faces) {
    faceCount.textContent = faces.length;
    faceEmpty.hidden = faces.length > 0;
    const seen = new Set();
    faces.forEach(f => {
      const key = String(f.track_id ?? `${f.identity}-${f.box[0]}`);
      seen.add(key);
      let li = faceList.querySelector(`[data-key="${key}"]`);
      if (!li || li.dataset.identity !== f.identity) {
        const fresh = document.createElement('li');
        fresh.className = 'face-item';
        fresh.dataset.key = key;
        fresh.dataset.identity = f.identity;
        fresh.innerHTML = `${avatar(f.identity)}<div class="face-info"><span class="face-name">${esc(displayName(f))}</span><div class="face-meta"></div></div>`;
        li ? li.replaceWith(fresh) : faceList.appendChild(fresh);
        li = fresh;
      }
      const meta = li.querySelector('.face-info');
      const html = `<span class="face-name">${esc(displayName(f))}</span><div class="face-meta">${metaHtml(f)}</div>`;
      if (meta.dataset.html !== html) { meta.innerHTML = html; meta.dataset.html = html; }
    });
    faceList.querySelectorAll('.face-item').forEach(li => { if (!seen.has(li.dataset.key)) li.remove(); });
  }

  // Journal + bannière de bienvenue
  let bannerTimer = null;
  function addEvent(f, { fromHistory = false } = {}) {
    const time = fromHistory ? f.timestamp : new Date().toISOString();
    const li = document.createElement('li');
    li.className = 'event-item';
    li.innerHTML = `${avatar(f.identity)}<div class="face-info"><span class="face-name">${esc(f.identity)}</span>
      <span class="event-time">${fmtTime(time)} · ${Math.round(f.similarity * 100)}%</span></div>`;
    fromHistory ? eventList.appendChild(li) : eventList.prepend(li);
    while (eventList.children.length > 12) eventList.lastElementChild.remove();
    eventEmpty.hidden = true;
    if (fromHistory) return;

    $('welcome-avatar').innerHTML = avatar(f.identity);
    $('welcome-name').textContent = `Bonjour ${f.identity} !`;
    $('welcome-sub').textContent = `Pointage enregistré à ${fmtTime(time)}`;
    banner.classList.add('show');
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => banner.classList.remove('show'), 3500);
  }

  async function loadTodayEvents() {
    try {
      const r = await fetch('/api/logs?limit=40');
      const { logs } = await r.json();
      logs.filter(l => isToday(l.timestamp) && !/SPOOF|FRAUDE/i.test(l.liveness_status || ''))
        .slice(0, 12)
        .forEach(l => addEvent({ identity: l.user_name, similarity: Number(l.confidence_score) || 0, timestamp: l.timestamp }, { fromHistory: true }));
    } catch { /* le journal se remplira en direct */ }
  }

  // --- Contrôles --------------------------------------------------------------
  function setPaused(p) {
    state.paused = p;
    btnPause.setAttribute('aria-pressed', String(p));
    btnPause.querySelector('use').setAttribute('href', p ? '#i-play' : '#i-pause');
    btnPause.setAttribute('aria-label', p ? 'Reprendre' : 'Mettre en pause');
    if (p) { ctx.clearRect(0, 0, canvas.width, canvas.height); setLive('paused', 'En pause'); video.pause(); }
    else { video.play(); if (state.wsOpen) setLive('ok', 'Analyse en direct'); scheduleSend(0); }
  }
  function setMirror(m) {
    state.mirrored = m; savePref('mirror', m);
    stage.classList.toggle('mirrored', m);
    btnMirror.setAttribute('aria-pressed', String(m));
    draw(state.lastFaces);
  }
  function setKps(v) {
    state.showKps = v; savePref('kps', v);
    btnKps.setAttribute('aria-pressed', String(v));
    draw(state.lastFaces);
  }
  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (stageCard.requestFullscreen) stageCard.requestFullscreen().catch(() => toast('Plein écran non disponible.', 'warn'));
  }
  document.addEventListener('fullscreenchange', () => {
    if (document.fullscreenElement) stage.style.aspectRatio = '';  // la vidéo remplit l'écran
    syncSize();
    draw(state.lastFaces);
  });

  btnPause.addEventListener('click', () => setPaused(!state.paused));
  btnMirror.addEventListener('click', () => setMirror(!state.mirrored));
  btnKps.addEventListener('click', () => setKps(!state.showKps));
  btnFull.addEventListener('click', toggleFullscreen);
  document.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, button') && e.key === ' ') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === ' ') { e.preventDefault(); setPaused(!state.paused); }
    else if (k === 'm') setMirror(!state.mirrored);
    else if (k === 'p') setKps(!state.showKps);
    else if (k === 'f') toggleFullscreen();
  });

  // Onglet masqué : on arrête d'envoyer des images (économie CPU), reprise automatique.
  document.addEventListener('visibilitychange', () => { if (!document.hidden) scheduleSend(0); });
  // Libère la webcam en quittant la page (sinon la page Enrôler ne peut pas l'ouvrir).
  window.addEventListener('pagehide', stopCamera);

  // --- Démarrage ----------------------------------------------------------------
  setMirror(state.mirrored);
  setKps(state.showKps);
  loadTodayEvents();
  startCamera();
  connect();
})();
