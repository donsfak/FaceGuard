// dashboard.js — tableau de bord en direct (rafraîchi toutes les 5 s via /api/logs).
(() => {
  'use strict';
  const { esc, avatar, statusBadge, fmtTime, fmtRelative, fmtDateTime, isToday } = window.FG;
  const $ = id => document.getElementById(id);
  const REFRESH_MS = 5000;

  const state = {
    data: JSON.parse($('initial-data').textContent),
    filter: 'all', query: '', shown: 15, knownKeys: null, lastUpdate: Date.now(), failed: false,
  };

  const isAlert = l => /SPOOF|FRAUDE/i.test(l.liveness_status || '');
  const keyOf = l => `${l.timestamp}|${l.user_name}`;

  function renderStats() {
    const { stats, logs } = state.data;
    const present = stats.present_today.length;
    $('stat-present').innerHTML = `${present} <small>/ ${stats.registered}</small>`;
    $('stat-present-meter').style.width = stats.registered ? `${(present / stats.registered) * 100}%` : '0';
    $('stat-today').textContent = stats.today;
    $('stat-today-foot').textContent = stats.today ? `${present} personne${present > 1 ? 's' : ''} différente${present > 1 ? 's' : ''}` : 'aucun passage';
    const last = logs.find(l => !isAlert(l));
    $('stat-last').textContent = last ? last.user_name : '—';
    $('stat-last-foot').textContent = last ? fmtRelative(last.timestamp) : '';
    $('stat-alerts').textContent = stats.alerts;
    $('stat-alerts-card').classList.toggle('alert', stats.alerts > 0);
    $('source-text').textContent = state.data.source;
    const sbOk = state.data.source === 'Supabase';
    $('source-chip').title = sbOk ? 'Données synchronisées avec Supabase'
      : `Pointages enregistrés localement (docs/recognition_log.csv). Raison : ${state.data.supabase_status || 'inconnue'}`;
    const banner = $('sb-banner');
    banner.hidden = sbOk;
    if (!sbOk) $('sb-reason').textContent = state.data.supabase_status || 'raison inconnue';
  }

  function renderLogs() {
    const q = state.query.toLowerCase();
    const rows = state.data.logs.filter(l =>
      (state.filter === 'all' || (state.filter === 'alert') === isAlert(l)) &&
      (!q || String(l.user_name || '').toLowerCase().includes(q)));
    const fresh = state.knownKeys;
    const more = $('log-more');
    $('log-more-wrap').hidden = rows.length <= state.shown;
    more.textContent = `Afficher plus (${rows.length - state.shown} restants)`;
    $('log-body').innerHTML = rows.slice(0, state.shown).map(l => {
      const score = Math.max(0, Math.min(1, Number(l.confidence_score) || 0));
      const isNew = fresh && !fresh.has(keyOf(l));
      return `<tr class="${isNew ? 'row-new' : ''}">
        <td><div class="person-cell">${avatar(l.user_name)}<span>${esc(l.user_name)}</span></div></td>
        <td class="col-status">${statusBadge(l.liveness_status)}</td>
        <td class="col-score"><div class="score"><div class="score-bar"><span style="width:${score * 100}%"></span></div><span class="score-val">${Math.round(score * 100)}%</span></div></td>
        <td class="col-time time-cell" title="${esc(fmtDateTime(l.timestamp))}">${esc(fmtRelative(l.timestamp))}<span class="abs">${esc(isToday(l.timestamp) ? fmtTime(l.timestamp) : fmtDateTime(l.timestamp))}</span></td>
      </tr>`;
    }).join('');
    state.knownKeys = new Set(state.data.logs.map(keyOf));
    $('journal-count').textContent = rows.length;
    const empty = rows.length === 0;
    $('log-empty').hidden = !empty;
    if (empty) {
      const filtered = state.data.logs.length > 0;
      $('empty-title').textContent = filtered ? 'Aucun résultat' : 'Aucun pointage pour le moment';
      $('empty-text').textContent = filtered ? 'Modifiez la recherche ou le filtre.' : 'Ouvrez le scanner et passez devant la caméra.';
    }
  }

  // Présence du jour : heure d'arrivée = premier pointage vérifié de la journée
  function renderPresence() {
    const { stats, logs } = state.data;
    const arrival = {};
    logs.filter(l => isToday(l.timestamp) && !isAlert(l)).forEach(l => { arrival[l.user_name] = l.timestamp; });
    const presentHtml = stats.present_today
      .sort((a, b) => new Date(arrival[a]) - new Date(arrival[b]))
      .map(n => `<li class="presence-item">${avatar(n)}<div class="face-info"><span class="face-name">${esc(n)}</span>
        <span class="event-time">arrivé(e) à ${fmtTime(arrival[n])}</span></div><span class="badge badge-ok">Présent</span></li>`).join('');
    const absentHtml = stats.absent_today
      .map(n => `<li class="presence-item absent">${avatar(n)}<div class="face-info"><span class="face-name">${esc(n)}</span>
        <span class="event-time">pas encore vu(e)</span></div></li>`).join('');
    $('presence').innerHTML = (stats.registered === 0)
      ? `<div class="empty"><strong>Aucune personne enregistrée</strong><a class="btn btn-primary" href="/register">Enrôler une personne</a></div>`
      : `${presentHtml ? `<p class="presence-section">Présents · ${stats.present_today.length}</p><ul class="presence-list">${presentHtml}</ul>` : ''}
         ${absentHtml ? `<p class="presence-section">Absents · ${stats.absent_today.length}</p><ul class="presence-list">${absentHtml}</ul>` : ''}`;
  }

  function renderAll() { renderStats(); renderLogs(); renderPresence(); }

  function renderUpdated() {
    const s = Math.round((Date.now() - state.lastUpdate) / 1000);
    $('updated').classList.toggle('stale', state.failed);
    $('updated-text').textContent = state.failed ? 'Hors connexion — nouvelle tentative…'
      : s < 3 ? 'À jour' : `Mis à jour il y a ${s} s`;
  }

  async function refresh() {
    if (document.hidden) return;
    try {
      const r = await fetch('/api/logs?limit=100', { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      state.data = await r.json();
      state.lastUpdate = Date.now();
      state.failed = false;
      renderAll();
    } catch {
      state.failed = true;
    }
    renderUpdated();
  }

  $('search').addEventListener('input', e => { state.query = e.target.value.trim(); state.shown = 15; renderLogs(); });
  $('log-more').addEventListener('click', () => { state.shown += 25; renderLogs(); });
  document.querySelectorAll('[data-filter]').forEach(b => b.addEventListener('click', () => {
    state.filter = b.dataset.filter;
    state.shown = 15;
    document.querySelectorAll('[data-filter]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    renderLogs();
  }));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });

  renderAll();
  renderUpdated();
  setInterval(refresh, REFRESH_MS);
  setInterval(renderUpdated, 1000);
})();
