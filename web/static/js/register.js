// register.js — enrôlement d'une personne avec 8 à 10 photos.
// Deux sources, combinables dans la même limite de 10 photos :
//  • caméra : 8 poses guidées (+ 2 facultatives). Contrôle qualité en continu
//    (/api/check-face) ; chaque photo n'est prise que lorsque le visage est bien
//    cadré, net et éclairé, et reste stable.
//  • import : photos choisies dans les fichiers de l'utilisateur. Chacune est
//    redimensionnée dans le navigateur puis vérifiée (/api/check-face, mode
//    "import") : une photo sans visage, floue ou avec plusieurs personnes est
//    refusée tout de suite, avec la raison.
// Enfin envoi à /register-user, qui vérifie aussi que toutes les photos
// montrent la même personne.
(() => {
  'use strict';
  const { esc, icon, toast } = window.FG;
  const $ = id => document.getElementById(id);

  const panel = document.querySelector('.enroll-panel');
  const MIN_PHOTOS = Number(panel.dataset.minPhotos) || 8;
  const MAX_PHOTOS = Number(panel.dataset.maxPhotos) || 10;
  const KNOWN_PEOPLE = (panel.dataset.people || '').split('|').filter(Boolean);

  // Les MIN_PHOTOS premières poses sont guidées ; les suivantes sont facultatives.
  const POSES = [
    { short: 'Face', text: 'Regardez droit vers la caméra' },
    { short: 'Gauche', text: 'Tournez légèrement la tête vers votre gauche' },
    { short: 'Droite', text: 'Tournez légèrement la tête vers votre droite' },
    { short: 'Menton haut', text: 'Levez légèrement le menton' },
    { short: 'Menton bas', text: 'Baissez légèrement le menton' },
    { short: 'Sourire', text: 'Regardez la caméra et souriez' },
    { short: 'Penchée', text: 'Penchez légèrement la tête sur le côté' },
    { short: 'Proche', text: 'Rapprochez-vous un peu de la caméra' },
    { short: 'Neutre', text: 'Regardez la caméra, expression neutre' },
    { short: 'Libre', text: 'Changez légèrement de position' },
  ].slice(0, MAX_PHOTOS);
  const NAME_RE = /^[\p{L}\p{N}_\- '.]+$/u;
  const STABLE_CHECKS = 2;      // vérifications "ok" consécutives avant la photo
  const CHECK_PAUSE_MS = 250;   // pause entre deux vérifications
  const IMPORT_MAX_SIDE = 1600; // les photos importées sont réduites avant l'envoi
  const MAX_FILE_BYTES = 25 * 1024 * 1024;

  const els = {
    stage: $('stage'), video: $('video'), guide: $('face-guide'), flash: $('flash'),
    placeholder: $('stage-placeholder'), placeholderText: $('placeholder-text'), retry: $('camera-retry'),
    pill: $('quality-pill'), pillText: $('quality-text'),
    instruction: $('instruction'), instructionStep: $('instruction-step'), instructionText: $('instruction-text'), countdown: $('countdown'),
    switchBtn: $('switch-camera'),
    form: $('enroll-form'), name: $('name'), nameField: $('name-field'), nameLen: $('name-len'), nameError: $('name-error'),
    nameExists: $('name-exists'),
    consent: $('consent'), alert: $('alert'), thumbs: $('thumbs'), count: $('capture-count'),
    startBtn: $('start-btn'), startLabel: $('start-label'), startHint: $('start-hint'), cancelBtn: $('cancel-btn'),
    importBtn: $('import-btn'), importLabel: $('import-label'), importInput: $('import-input'),
    saveBtn: $('save-btn'), saveLabel: $('save-label'), restartBtn: $('restart-btn'),
    idle: $('actions-idle'), capturing: $('actions-capturing'), review: $('actions-review'),
    success: $('success'), successTitle: $('success-title'), successText: $('success-text'), another: $('another-btn'),
    steps: document.querySelectorAll('.step'),
  };

  const state = {
    stream: null, facing: 'user', phase: 'idle',   // idle | capturing | importing | sending | done
    photos: new Array(MAX_PHOTOS).fill(null),      // { blob, url, source: 'camera'|'import', mirror }
    lastCheck: null, okStreak: 0, checking: false, cancelToken: 0,
  };

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const count = () => state.photos.filter(Boolean).length;
  const freeSlots = () => state.photos.map((p, i) => (p ? -1 : i)).filter(i => i >= 0);
  const plural = (n, word) => `${n} ${word}${n > 1 ? 's' : ''}`;

  // ------------------------------------------------------------- caméra
  function cameraError(err) {
    switch (err && err.name) {
      case 'NotAllowedError': case 'SecurityError':
        return "Accès à la caméra refusé. Autorisez-la via l'icône à gauche de l'adresse du site, ou importez des photos.";
      case 'NotFoundError': case 'OverconstrainedError':
        return 'Aucune caméra détectée. Branchez une webcam, ou importez des photos.';
      case 'NotReadableError': case 'AbortError':
        return 'Caméra déjà utilisée (onglet Scanner, visio…). Fermez-la puis réessayez, ou importez des photos.';
      default:
        return window.isSecureContext ? `Caméra indisponible (${err && err.message}). Vous pouvez importer des photos.`
          : 'La caméra exige HTTPS (ou localhost). Vous pouvez importer des photos.';
    }
  }

  const cameraReady = () => !!(state.stream && state.stream.active);

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
      updateActions();
      qualityLoop();
    } catch (err) {
      console.error(err);
      state.stream = null;
      els.stage.classList.add('no-camera');
      els.placeholderText.textContent = cameraError(err);
      els.retry.hidden = false;
      updateActions();
    }
  }
  els.retry.addEventListener('click', () => { els.stage.classList.remove('no-camera'); startCamera(); });
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
    while (cameraReady() && ['idle', 'capturing'].includes(state.phase)) {
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

  // Prévient AVANT l'enregistrement si ce visage est déjà connu sous un autre nom.
  function looksLikeSomeoneElse(res) {
    const typed = els.name.value.trim().toLowerCase();
    return res.status === 'ok' && res.known_as && res.similarity >= 0.55 && res.known_as.toLowerCase() !== typed;
  }
  function showKnownWarning(res) {
    if (state.phase !== 'idle' && state.phase !== 'capturing') return;
    if (els.alert.dataset.sticky) return;   // ne pas masquer un bilan d'import ou une erreur
    if (looksLikeSomeoneElse(res)) {
      showAlert(`Ce visage ressemble fortement à « ${res.known_as} », déjà enregistré(e). Vérifiez qu'il ne s'agit pas de la même personne.`, 'warn');
    } else if (els.alert.dataset.kind === 'warn' && !els.alert.dataset.sticky) {
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
  const existingName = () => {
    const v = els.name.value.trim().toLowerCase();
    return KNOWN_PEOPLE.find(p => p.toLowerCase() === v) || null;
  };
  const formReady = () => !nameError() && els.consent.checked;

  els.name.addEventListener('input', () => {
    els.nameLen.textContent = els.name.value.length;
    if (els.nameField.classList.contains('invalid')) validateName();
    const known = existingName();
    els.nameExists.hidden = !known;
    if (known) els.nameExists.textContent = `« ${known} » est déjà enregistré(e) : ces photos compléteront son profil.`;
    updateActions();
  });
  els.name.addEventListener('blur', () => { if (els.name.value) validateName(); });
  els.consent.addEventListener('change', updateActions);
  function validateName() {
    const err = nameError();
    els.nameField.classList.toggle('invalid', !!err);
    els.nameError.textContent = err;
    return !err;
  }

  function showAlert(message, kind = 'error', { sticky = false, html = false } = {}) {
    els.alert.innerHTML = `${icon('i-alert')}<div>${html ? message : esc(message)}</div>`;
    els.alert.className = `alert ${kind === 'warn' ? 'warn' : ''}`;
    els.alert.dataset.kind = kind;
    els.alert.dataset.sticky = sticky ? '1' : '';
    els.alert.hidden = false;
  }
  function hideAlert() { els.alert.hidden = true; els.alert.dataset.kind = ''; els.alert.dataset.sticky = ''; }

  function rejectionList(title, items) {
    return `<strong>${esc(title)}</strong><ul class="reject-list">${
      items.map(r => `<li><span>${esc(r.file)}</span> : ${esc(r.reason)}</li>`).join('')}</ul>`;
  }

  // -------------------------------------------------------------- vignettes
  function renderThumbs(current = -1) {
    const n = count();
    els.count.textContent = `${n}/${MAX_PHOTOS} · minimum ${MIN_PHOTOS}`;
    const editable = state.phase === 'idle';
    els.thumbs.innerHTML = POSES.map((p, i) => {
      const photo = state.photos[i];
      const optional = i >= MIN_PHOTOS;
      const label = photo && photo.source === 'import' ? 'Importée' : (optional ? `${p.short} (facult.)` : p.short);
      const remove = photo && editable
        ? `<button type="button" class="thumb-remove" data-remove="${i}" title="Retirer" aria-label="Retirer la photo ${i + 1}">${icon('i-x')}</button>` : '';
      const img = photo
        ? `<img src="${photo.url}" alt="Photo ${i + 1} : ${esc(label)}" class="${photo.mirror ? 'mirror' : ''}">` : i + 1;
      return `<li class="thumb ${photo ? 'filled' : ''} ${optional ? 'optional' : ''} ${i === current ? 'current' : ''}">
        <div class="thumb-frame">${img}${remove}</div>
        <span class="thumb-label">${esc(label)}</span></li>`;
    }).join('');
  }
  els.thumbs.addEventListener('click', e => {
    const b = e.target.closest('[data-remove]');
    if (!b || state.phase !== 'idle') return;
    setPhoto(Number(b.dataset.remove), null);
    updateActions();
  });

  function setPhoto(i, photo) {
    const old = state.photos[i];
    if (old) URL.revokeObjectURL(old.url);
    state.photos[i] = photo ? { ...photo, url: URL.createObjectURL(photo.blob) } : null;
  }

  // --------------------------------------------------------- boutons / étapes
  function updateActions() {
    const n = count();
    const free = MAX_PHOTOS - n;
    const guidedLeft = freeSlots().filter(i => i < MIN_PHOTOS).length;
    const ready = formReady();
    const busy = state.phase !== 'idle';

    els.startBtn.disabled = busy || !ready || !cameraReady() || free === 0;
    els.importBtn.disabled = busy || !ready || free === 0;
    els.startLabel.textContent = n === 0 ? `Prendre ${MIN_PHOTOS} photos guidées`
      : guidedLeft > 0 ? `Compléter avec la caméra (${guidedLeft})`
      : free > 0 ? 'Ajouter une photo (caméra)' : 'Caméra';
    els.importLabel.textContent = n > 0 && free > 0 ? `Importer des photos (${free} max)` : 'Importer des photos';

    // Une fois le minimum atteint, l'enregistrement devient l'action principale.
    const enough = n >= MIN_PHOTOS;
    els.startBtn.classList.toggle('btn-primary', !enough);
    els.startBtn.classList.toggle('btn-secondary', enough);

    els.startHint.textContent = !ready ? "Saisissez un nom et cochez l'accord pour commencer."
      : n === 0 ? `Prenez ${MIN_PHOTOS} photos guidées avec la caméra, importez les vôtres, ou combinez les deux (${MIN_PHOTOS} à ${MAX_PHOTOS} photos).`
      : n < MIN_PHOTOS ? `Encore ${plural(MIN_PHOTOS - n, 'photo')} pour atteindre le minimum de ${MIN_PHOTOS}.`
      : free > 0 ? `Vous pouvez enregistrer, ou ajouter jusqu'à ${plural(free, 'photo')} de plus.`
      : `Limite de ${MAX_PHOTOS} photos atteinte : retirez-en une pour la remplacer.`;
    if (!cameraReady() && ready && n < MAX_PHOTOS && state.phase === 'idle') {
      els.startHint.textContent += ' Caméra indisponible : utilisez l\'import.';
    }

    els.review.hidden = n === 0 || state.phase === 'capturing' || state.phase === 'done';
    els.saveBtn.disabled = !enough || !ready || busy;
    if (state.phase !== 'sending') {
      els.saveLabel.textContent = enough ? `Enregistrer ${els.name.value.trim()} · ${plural(n, 'photo')}`
        : `Enregistrer (${n}/${MIN_PHOTOS} minimum)`;
    }
    renderSteps();
    renderThumbs();
  }

  function renderSteps() {
    const n = count();
    const order = state.phase === 'done' ? 3
      : n >= MIN_PHOTOS ? 2
      : (n > 0 || state.phase === 'capturing' || state.phase === 'importing') ? 1 : 0;
    els.steps.forEach((s, i) => {
      s.classList.toggle('active', i === order);
      s.classList.toggle('done', i < order);
    });
  }

  function setPhase(phase) {
    state.phase = phase;
    els.idle.hidden = phase === 'capturing' || phase === 'done';
    els.capturing.hidden = phase !== 'capturing';
    els.form.hidden = phase === 'done';
    els.success.hidden = phase !== 'done';
    els.name.disabled = els.consent.disabled = ['capturing', 'importing', 'sending'].includes(phase);
    els.instruction.hidden = phase !== 'capturing';
    updateActions();
    if (['idle', 'capturing'].includes(phase) && cameraReady()) qualityLoop();
  }

  // ------------------------------------------------------- photos caméra
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
      for (const [k, i] of indices.entries()) {
        renderThumbs(i);
        els.instructionStep.textContent = `Photo ${i + 1} · ${indices.length - k} restante${indices.length - k > 1 ? 's' : ''}`;
        els.instructionText.textContent = POSES[i].text;
        els.countdown.textContent = '';
        await sleep(900);                       // laisse le temps de changer de pose
        await waitForGoodFrame(token);
        for (const n of [3, 2, 1]) {            // petit compte à rebours : on ne bouge plus
          if (token !== state.cancelToken) throw new Error('cancelled');
          els.countdown.textContent = `Ne bougez plus… ${n}`;
          await sleep(350);
        }
        if (state.lastCheck && state.lastCheck.status !== 'ok') await waitForGoodFrame(token);
        const blob = await grab(1280, 0.92);
        els.flash.classList.remove('go'); void els.flash.offsetWidth; els.flash.classList.add('go');
        setPhoto(i, { blob, source: 'camera', mirror: state.facing === 'user' });
        renderThumbs(i);
      }
    } catch (e) {
      if (e.message !== 'cancelled') throw e;
    }
    if (token === state.cancelToken) setPhase('idle');
  }

  els.form.addEventListener('submit', e => {
    e.preventDefault();
    if (!validateName()) { els.name.focus(); return; }
    if (!els.consent.checked) { showAlert("L'accord de la personne est obligatoire."); return; }
    const free = freeSlots();
    if (!free.length) return;
    // D'abord les poses guidées manquantes ; au-delà, une photo facultative à la fois.
    const guided = free.filter(i => i < MIN_PHOTOS);
    runCaptures(guided.length ? guided : [free[0]]);
  });

  els.cancelBtn.addEventListener('click', () => { state.cancelToken++; setPhase('idle'); });

  // ------------------------------------------------------- photos importées
  async function loadImage(file) {
    if (window.createImageBitmap) {
      try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch { /* repli */ }
    }
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally { URL.revokeObjectURL(url); }
  }

  // Réduit la photo (plus légère à envoyer, orientation EXIF appliquée) et la convertit en JPEG.
  async function normalizeImport(file) {
    const img = await loadImage(file);
    const w = img.width, h = img.height;
    const scale = Math.min(1, IMPORT_MAX_SIDE / Math.max(w, h));
    const c = document.createElement('canvas');
    c.width = Math.round(w * scale); c.height = Math.round(h * scale);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    if (img.close) img.close();
    return new Promise((res, rej) => c.toBlob(b => (b ? res(b) : rej(new Error('conversion'))), 'image/jpeg', 0.92));
  }

  els.importBtn.addEventListener('click', () => {
    if (!validateName()) { els.name.focus(); return; }
    if (!els.consent.checked) { showAlert("L'accord de la personne est obligatoire."); return; }
    els.importInput.value = '';
    els.importInput.click();
  });

  els.importInput.addEventListener('change', async () => {
    const files = Array.from(els.importInput.files || []);
    if (!files.length) return;
    hideAlert();
    const slots = freeSlots();
    const rejected = [];
    let added = 0, processed = 0, warnKnown = null;

    setPhase('importing');
    els.importBtn.disabled = els.startBtn.disabled = true;
    // Une photo refusée libère sa place pour le fichier suivant : on analyse
    // jusqu'à remplir les places libres, les fichiers restants sont ignorés.
    for (const file of files) {
      if (added >= slots.length) break;
      processed++;
      els.importLabel.textContent = `Analyse ${processed}/${files.length}…`;
      try {
        if (!/^image\/(jpeg|png|webp)$/.test(file.type)) throw new Error('format non pris en charge (JPEG, PNG ou WebP)');
        if (file.size > MAX_FILE_BYTES) throw new Error('fichier trop lourd (plus de 25 Mo)');
        let blob;
        try { blob = await normalizeImport(file); } catch { throw new Error('image illisible'); }
        const fd = new FormData();
        fd.append('file', blob, 'import.jpg');
        fd.append('mode', 'import');
        const r = await fetch('/api/check-face', { method: 'POST', body: fd });
        if (!r.ok) throw new Error('image illisible');
        const res = await r.json();
        if (res.status !== 'ok') throw new Error(res.message.replace(/\.$/, '').toLowerCase());
        if (looksLikeSomeoneElse(res)) warnKnown = res;
        setPhoto(slots[added], { blob, source: 'import', mirror: false });
        added++;
        renderThumbs();
      } catch (err) {
        rejected.push({ file: file.name, reason: err.message || 'erreur inconnue' });
      }
    }
    setPhase('idle');
    const ignored = files.length - processed;

    const parts = [];
    if (rejected.length) parts.push(rejectionList(`${plural(rejected.length, 'photo')} refusée${rejected.length > 1 ? 's' : ''} :`, rejected));
    if (ignored) parts.push(`<p>${plural(ignored, 'photo')} non importée${ignored > 1 ? 's' : ''} : limite de ${MAX_PHOTOS} photos.</p>`);
    if (warnKnown) parts.push(`<p>Un visage importé ressemble fortement à « ${esc(warnKnown.known_as)} », déjà enregistré(e).</p>`);
    if (parts.length) showAlert(parts.join(''), 'warn', { sticky: true, html: true });
    if (added) toast(`${plural(added, 'photo')} importée${added > 1 ? 's' : ''}.`, 'success');
  });

  // ------------------------------------------------------------ enregistrement
  function resetAll() {
    state.cancelToken++;
    state.photos.forEach((_, i) => setPhoto(i, null));
    hideAlert();
    setPhase('idle');
  }
  els.restartBtn.addEventListener('click', resetAll);

  els.saveBtn.addEventListener('click', async () => {
    const name = els.name.value.trim();
    const fd = new FormData();
    fd.append('name', name);
    state.photos.forEach((p, i) => { if (p) fd.append('files', p.blob, `${p.source}_${i + 1}.jpg`); });
    hideAlert();
    setPhase('sending');
    els.saveLabel.textContent = 'Analyse des photos…';
    try {
      const r = await fetch('/register-user', { method: 'POST', body: fd });
      let data = {};
      try { data = await r.json(); } catch { /* réponse non JSON */ }
      if (!r.ok) {
        const d = data.detail;
        const err = new Error(Array.isArray(d) ? d.map(x => x.msg).join(' ; ')
          : (d && typeof d === 'object') ? d.message : (d || `Erreur serveur (${r.status})`));
        err.rejected = d && d.rejected;
        throw err;
      }
      els.successTitle.textContent = data.updated ? `Profil de ${data.name} complété` : `${data.name} est enregistré(e)`;
      let text = `${data.message} La personne peut maintenant être reconnue par le scanner.`;
      if (data.rejected && data.rejected.length) text += ` ${plural(data.rejected.length, 'photo')} écartée${data.rejected.length > 1 ? 's' : ''} à l'analyse.`;
      els.successText.textContent = text;
      stopTracks();
      els.stage.classList.add('finished');
      els.placeholder.hidden = false; els.retry.hidden = true;
      els.placeholderText.textContent = `${data.name} fait maintenant partie de la base de reconnaissance.`;
      if (!KNOWN_PEOPLE.some(p => p.toLowerCase() === data.name.toLowerCase())) KNOWN_PEOPLE.push(data.name);
      setPhase('done');
      toast(`${data.name} ajouté(e) à la base.`, 'success');
    } catch (err) {
      setPhase('idle');
      if (err.rejected && err.rejected.length) {
        showAlert(`<p>${esc(err.message)}</p>${rejectionList('Photos écartées :', err.rejected)}`, 'error', { sticky: true, html: true });
      } else {
        showAlert(err.message, 'error', { sticky: true });
      }
    }
  });

  els.another.addEventListener('click', () => {
    els.stage.classList.remove('finished', 'no-camera');
    resetAll();
    els.form.reset();
    els.nameLen.textContent = '0';
    els.nameExists.hidden = true;
    startCamera();
    els.name.focus();
  });

  // ---------------------------------------------------------------- démarrage
  setPhase('idle');
  startCamera();
})();
