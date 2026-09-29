// register.js — enrôlement guidé.
// 1. Contrôle qualité en continu (/api/check-face) : un seul visage, assez grand, net, éclairé.
// 2. 5 photos avec des poses différentes (variété = meilleure reconnaissance),
//    chaque photo n'est prise que lorsque la qualité est bonne et stable.
// 3. Relecture (possibilité de reprendre une photo), puis envoi à /register-user.
(() => {
  'use strict';
  const { esc, icon, toast } = window.FG;
  const $ = id => document.getElementById(id);

  const POSES = [
    { short: 'Face', text: 'Regardez droit vers la caméra' },
    { short: 'Gauche', text: 'Tournez légèrement la tête vers votre gauche' },
    { short: 'Droite', text: 'Tournez légèrement la tête vers votre droite' },
    { short: 'Menton', text: 'Levez légèrement le menton' },
    { short: 'Sourire', text: 'Regardez la caméra et souriez' },
  ];
  const NAME_RE = /^[\p{L}\p{N}_\- '.]+$/u;
  const STABLE_CHECKS = 2;     // vérifications "ok" consécutives avant la photo
  const CHECK_PAUSE_MS = 250;  // pause entre deux vérifications

  const els = {
    stage: $('stage'), video: $('video'), guide: $('face-guide'), flash: $('flash'),
    placeholder: $('stage-placeholder'), placeholderText: $('placeholder-text'), retry: $('camera-retry'),
    pill: $('quality-pill'), pillText: $('quality-text'),
    instruction: $('instruction'), instructionStep: $('instruction-step'), instructionText: $('instruction-text'), countdown: $('countdown'),
    switchBtn: $('switch-camera'),
    form: $('enroll-form'), name: $('name'), nameField: $('name-field'), nameLen: $('name-len'), nameError: $('name-error'),
    consent: $('consent'), alert: $('alert'), thumbs: $('thumbs'), count: $('capture-count'),
    startBtn: $('start-btn'), startHint: $('start-hint'), cancelBtn: $('cancel-btn'),
    saveBtn: $('save-btn'), saveLabel: $('save-label'), restartBtn: $('restart-btn'),
    idle: $('actions-idle'), capturing: $('actions-capturing'), review: $('actions-review'),
    success: $('success'), successTitle: $('success-title'), successText: $('success-text'), another: $('another-btn'),
    steps: document.querySelectorAll('.step'),
  };

  const state = {
    stream: null, facing: 'user', phase: 'idle',
    captures: new Array(POSES.length).fill(null),   // Blob par pose
    urls: new Array(POSES.length).fill(null),
    lastCheck: null, okStreak: 0, checking: false, cancelToken: 0,
  };

  // ------------------------------------------------------------- caméra
  function cameraError(err) {
    switch (err && err.name) {
      case 'NotAllowedError': case 'SecurityError':
        return "Accès à la caméra refusé. Autorisez-la via l'icône à gauche de l'adresse du site.";
      case 'NotFoundError': case 'OverconstrainedError':
        return 'Aucune caméra détectée. Branchez une webcam puis réessayez.';
      case 'NotReadableError': case 'AbortError':
        return 'Caméra déjà utilisée (onglet Scanner, visio…). Fermez-la puis réessayez.';
      default:
        return window.isSecureContext ? `Caméra indisponible (${err && err.message})` : 'La caméra exige HTTPS (ou localhost).';
    }
  }

  async function startCamera() {
    if (state.stream) state.stream.getTracks().forEach(t => t.stop());
    els.placeholder.hidden = false; els.retry.hidden = true;
    els.placeholderText.textContent = 'Activation de la caméra…';
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      els.video.srcObject = state.stream;
      await els.video.play();
      els.stage.style.aspectRatio = `${els.video.videoWidth} / ${els.video.videoHeight}`;
      els.stage.classList.toggle('mirrored', state.facing === 'user');
      els.placeholder.hidden = true;
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      els.switchBtn.hidden = cams.length < 2;
      updateStartButton();
      qualityLoop();
    } catch (err) {
      console.error(err);
      els.placeholderText.textContent = cameraError(err);
      els.retry.hidden = false;
      updateStartButton();
    }
  }
  els.retry.addEventListener('click', startCamera);
  els.switchBtn.addEventListener('click', () => { state.facing = state.facing === 'user' ? 'environment' : 'user'; startCamera(); });
  const stopTracks = () => { if (state.stream) state.stream.getTracks().forEach(t => t.stop()); };
  window.addEventListener('pagehide', stopTracks);

  function grab(maxWidth, quality) {
    const v = els.video;
    const scale = Math.min(1, maxWidth / v.videoWidth);
    const c = document.createElement('canvas');
    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    return new Promise(res => c.toBlob(res, 'image/jpeg', quality));
  }

  // ------------------------------------------------------ contrôle qualité
  function setQuality(kind, text) {
    els.pill.className = `quality-pill ${kind}`;
    els.pillText.textContent = text;
    els.guide.className = `face-guide ${kind}`;
  }

  async function qualityLoop() {
    if (state.checking) return;
    state.checking = true;
    while (state.stream && state.stream.active && state.phase !== 'done' && state.phase !== 'sending') {
      if (document.hidden || !els.video.videoWidth) { await sleep(500); continue; }
      try {
        const fd = new FormData();
        fd.append('file', await grab(480, 0.8), 'check.jpg');
        const r = await fetch('/api/check-face', { method: 'POST', body: fd });
        const res = await r.json();
        state.lastCheck = res;
        state.okStreak = res.status === 'ok' ? state.okStreak + 1 : 0;
        setQuality(res.status === 'ok' ? 'ok' : 'bad', res.message);
        showKnownWarning(res);
      } catch {
        state.okStreak = 0;
        setQuality('bad', 'Serveur injoignable, nouvelle tentative…');
        await sleep(1500);
      }
      await sleep(CHECK_PAUSE_MS);
    }
    state.checking = false;
  }

  // Prévient AVANT les photos si ce visage est déjà enregistré sous un autre nom.
  function showKnownWarning(res) {
    if (state.phase === 'review' || state.phase === 'done') return;
    const typed = els.name.value.trim().toLowerCase();
    if (res.status === 'ok' && res.known_as && res.similarity >= 0.55 && res.known_as.toLowerCase() !== typed) {
      showAlert(`Ce visage ressemble fortement à « ${res.known_as} », déjà enregistré(e). Vérifiez qu'il ne s'agit pas de la même personne.`, 'warn');
    } else if (els.alert.dataset.kind === 'warn') {
      hideAlert();
    }
  }

  // ------------------------------------------------------------ formulaire
  function nameError() {
    const v = els.name.value.trim();
    if (!v) return 'Saisissez le nom de la personne.';
    if (!NAME_RE.test(v)) return 'Lettres, chiffres, espaces, tirets, apostrophes et points uniquement.';
    if (/^inconnu$/i.test(v)) return 'Ce nom est réservé par le système.';
    return '';
  }
  function updateStartButton() {
    const ready = !nameError() && els.consent.checked && state.stream && state.stream.active;
    els.startBtn.disabled = !ready;
    els.startHint.textContent = !state.stream ? 'La caméra doit être active.'
      : ready ? 'Les photos se prennent automatiquement quand le visage est bien cadré.'
      : "Saisissez un nom et cochez l'accord pour commencer.";
  }
  els.name.addEventListener('input', () => {
    els.nameLen.textContent = els.name.value.length;
    if (els.nameField.classList.contains('invalid')) validateName();
    updateStartButton();
  });
  els.name.addEventListener('blur', () => { if (els.name.value) validateName(); });
  els.consent.addEventListener('change', updateStartButton);
  function validateName() {
    const err = nameError();
    els.nameField.classList.toggle('invalid', !!err);
    els.nameError.textContent = err;
    return !err;
  }

  function showAlert(message, kind = 'error') {
    els.alert.innerHTML = `${icon('i-alert')}<span>${esc(message)}</span>`;
    els.alert.className = `alert ${kind === 'warn' ? 'warn' : ''}`;
    els.alert.dataset.kind = kind;
    els.alert.hidden = false;
  }
  function hideAlert() { els.alert.hidden = true; els.alert.dataset.kind = ''; }

  // -------------------------------------------------------------- vignettes
  function renderThumbs(current = -1) {
    const n = state.captures.filter(Boolean).length;
    els.count.textContent = `${n}/${POSES.length}`;
    els.thumbs.innerHTML = POSES.map((p, i) => {
      const filled = !!state.captures[i];
      const retake = filled && state.phase === 'review'
        ? `<button type="button" class="thumb-retake" data-retake="${i}" title="Reprendre" aria-label="Reprendre la photo ${i + 1}">${icon('i-refresh')}</button>` : '';
      return `<li class="thumb ${filled ? 'filled' : ''} ${i === current ? 'current' : ''}">
        <div class="thumb-frame">${filled ? `<img src="${state.urls[i]}" alt="Photo ${i + 1} : ${p.short}">` : i + 1}${retake}</div>
        <span class="thumb-label">${p.short}</span></li>`;
    }).join('');
  }
  els.thumbs.addEventListener('click', e => {
    const b = e.target.closest('[data-retake]');
    if (b) runCaptures([Number(b.dataset.retake)]);
  });

  // ---------------------------------------------------------------- étapes
  function setPhase(phase) {
    state.phase = phase;
    els.idle.hidden = phase !== 'idle';
    els.capturing.hidden = phase !== 'capturing';
    els.review.hidden = phase !== 'review' && phase !== 'sending';
    els.form.hidden = phase === 'done';
    els.success.hidden = phase !== 'done';
    els.name.disabled = els.consent.disabled = phase === 'capturing' || phase === 'sending';
    els.instruction.hidden = phase !== 'capturing';
    const order = { idle: 0, capturing: 1, review: 2, sending: 2, done: 3 };
    els.steps.forEach((s, i) => {
      s.classList.toggle('active', i === order[phase]);
      s.classList.toggle('done', i < order[phase]);
    });
    renderThumbs();
    // La boucle de contrôle qualité s'arrête pendant l'envoi : on la relance ensuite.
    if (['idle', 'capturing', 'review'].includes(phase) && state.stream && state.stream.active) qualityLoop();
  }

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function waitForGoodFrame(token) {
    state.okStreak = 0;
    while (state.okStreak < STABLE_CHECKS) {
      if (token !== state.cancelToken) throw new Error('cancelled');
      await sleep(120);
    }
  }

  async function runCaptures(indices) {
    hideAlert();
    const token = ++state.cancelToken;
    setPhase('capturing');
    try {
      for (const i of indices) {
        renderThumbs(i);
        els.instructionStep.textContent = `Photo ${i + 1} sur ${POSES.length}`;
        els.instructionText.textContent = POSES[i].text;
        els.countdown.textContent = '';
        await sleep(900);                       // laisse le temps de changer de pose
        await waitForGoodFrame(token);
        for (const n of [3, 2, 1]) {            // petit compte à rebours : on ne bouge plus
          if (token !== state.cancelToken) throw new Error('cancelled');
          els.countdown.textContent = `Ne bougez plus… ${n}`;
          await sleep(350);
        }
        if (state.lastCheck && state.lastCheck.status !== 'ok') { await waitForGoodFrame(token); }
        const blob = await grab(1280, 0.92);
        els.flash.classList.remove('go'); void els.flash.offsetWidth; els.flash.classList.add('go');
        if (state.urls[i]) URL.revokeObjectURL(state.urls[i]);
        state.captures[i] = blob;
        state.urls[i] = URL.createObjectURL(blob);
        renderThumbs(i);
      }
      const missing = state.captures.findIndex(c => !c);
      if (missing >= 0) return runCaptures(POSES.map((_, i) => i).filter(i => !state.captures[i]));
      setPhase('review');
      els.saveLabel.textContent = `Enregistrer ${els.name.value.trim()}`;
    } catch (e) {
      if (e.message !== 'cancelled') throw e;
    }
  }

  els.form.addEventListener('submit', e => {
    e.preventDefault();
    if (!validateName()) { els.name.focus(); return; }
    if (!els.consent.checked) { showAlert("L'accord de la personne est obligatoire."); return; }
    runCaptures(POSES.map((_, i) => i));
  });

  els.cancelBtn.addEventListener('click', () => {
    state.cancelToken++;
    const hasAll = state.captures.every(Boolean);
    setPhase(hasAll ? 'review' : 'idle');
  });

  function resetAll() {
    state.cancelToken++;
    state.urls.forEach(u => u && URL.revokeObjectURL(u));
    state.captures.fill(null); state.urls.fill(null);
    hideAlert();
    setPhase('idle');
  }
  els.restartBtn.addEventListener('click', resetAll);

  els.saveBtn.addEventListener('click', async () => {
    const name = els.name.value.trim();
    const fd = new FormData();
    fd.append('name', name);
    state.captures.forEach((b, i) => fd.append('files', b, `capture_${i}.jpg`));
    setPhase('sending');
    els.saveBtn.disabled = true;
    els.saveLabel.textContent = 'Analyse des photos…';
    try {
      const r = await fetch('/register-user', { method: 'POST', body: fd });
      let data = {};
      try { data = await r.json(); } catch { /* réponse non JSON */ }
      if (!r.ok) {
        const detail = Array.isArray(data.detail) ? data.detail.map(d => d.msg).join(' ; ') : data.detail;
        throw new Error(detail || `Erreur serveur (${r.status})`);
      }
      els.successTitle.textContent = `${data.name} est enregistré(e)`;
      els.successText.textContent = `${data.message} La personne peut maintenant être reconnue par le scanner.`;
      stopTracks();
      els.stage.classList.add('finished');
      els.placeholder.hidden = false; els.retry.hidden = true;
      els.placeholderText.textContent = `${data.name} fait maintenant partie de la base de reconnaissance.`;
      setPhase('done');
      toast(`${data.name} ajouté(e) à la base.`, 'success');
    } catch (err) {
      setPhase('review');
      showAlert(err.message);
    } finally {
      els.saveBtn.disabled = false;
      els.saveLabel.textContent = `Enregistrer ${name}`;
    }
  });

  els.another.addEventListener('click', () => {
    els.stage.classList.remove('finished');
    resetAll();
    els.form.reset();
    els.nameLen.textContent = '0';
    startCamera();
    els.name.focus();
  });

  // ---------------------------------------------------------------- démarrage
  setPhase('idle');
  startCamera();
})();
