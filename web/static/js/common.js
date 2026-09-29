// common.js — utilitaires partagés par toutes les pages FaceGuard.
(() => {
  'use strict';

  const ICONS = { info: 'i-dots', success: 'i-check', error: 'i-alert', warn: 'i-alert' };

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function icon(id, cls = '') {
    return `<svg class="icon ${cls}" aria-hidden="true"><use href="#${id}"/></svg>`;
  }

  // --- Notifications -------------------------------------------------------
  function toast(message, type = 'info', { duration = 4500 } = {}) {
    const stack = document.getElementById('toast-stack');
    if (!stack) return;
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.setAttribute('role', type === 'error' ? 'alert' : 'status');
    el.innerHTML = `${icon(ICONS[type] || ICONS.info)}<div class="toast-body">${esc(message)}</div>
      <button class="toast-close" type="button" aria-label="Fermer">${icon('i-x')}</button>`;
    const close = () => { el.classList.add('leaving'); setTimeout(() => el.remove(), 200); };
    el.querySelector('.toast-close').addEventListener('click', close);
    stack.appendChild(el);
    while (stack.children.length > 4) stack.firstElementChild.remove();
    if (duration) setTimeout(close, duration);
  }

  // --- Dates lisibles --------------------------------------------------------
  function parseDate(iso) {
    if (!iso) return null;
    const d = new Date(iso);
    return isNaN(d) ? null : d;
  }
  function fmtTime(iso) {
    const d = parseDate(iso);
    return d ? d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }) : '—';
  }
  function fmtDateTime(iso) {
    const d = parseDate(iso);
    return d ? d.toLocaleString('fr-FR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';
  }
  function isToday(iso) {
    const d = parseDate(iso);
    return !!d && d.toDateString() === new Date().toDateString();
  }
  function fmtRelative(iso) {
    const d = parseDate(iso);
    if (!d) return '—';
    const s = Math.round((Date.now() - d.getTime()) / 1000);
    if (s < 45) return "à l'instant";
    if (s < 3600) return `il y a ${Math.round(s / 60)} min`;
    if (isToday(iso)) return `il y a ${Math.floor(s / 3600)} h`;
    const yesterday = new Date(); yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) return `hier à ${fmtTime(iso)}`;
    return fmtDateTime(iso);
  }

  // --- Avatar : photo de la personne si disponible, sinon initiale ----------
  const UNKNOWN = 'Inconnu';
  const SPOOF = 'FRAUDE DETECTEE';
  function avatar(name, { size = '', photo = true } = {}) {
    const cls = name === UNKNOWN ? 'avatar-unknown' : name === SPOOF ? 'avatar-danger' : '';
    const letter = name === UNKNOWN ? '?' : name === SPOOF ? '!' : esc((name || '?').trim().charAt(0));
    const img = photo && name !== UNKNOWN && name !== SPOOF
      ? `<img src="/people/${encodeURIComponent(name)}/photo" alt="" loading="lazy" onerror="this.remove()">`
      : '';
    return `<span class="avatar ${size} ${cls}" aria-hidden="true">${letter}${img}</span>`;
  }

  function statusBadge(status) {
    const s = String(status || '').toUpperCase();
    if (s.includes('SPOOF') || s.includes('FRAUDE')) return `<span class="badge badge-danger">${icon('i-alert')}Fraude suspectée</span>`;
    if (s.includes('VIVANT')) return `<span class="badge badge-ok">${icon('i-shield')}Vérifié</span>`;
    if (s.includes('ANALYSE')) return `<span class="badge badge-info">Analyse…</span>`;
    return `<span class="badge">${esc(status || '—')}</span>`;
  }

  // --- État réel du serveur (remplace l'ancien "Système actif" figé) --------
  async function refreshHealth() {
    const nodes = document.querySelectorAll('.system-status');
    try {
      const r = await fetch('/health', { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const h = await r.json();
      nodes.forEach(n => {
        n.classList.add('ok'); n.classList.remove('down');
        n.querySelector('.system-status-text').textContent = n.classList.contains('compact')
          ? 'En ligne' : `IA en ligne · ${h.people} personne${h.people > 1 ? 's' : ''}`;
      });
    } catch {
      nodes.forEach(n => {
        n.classList.add('down'); n.classList.remove('ok');
        n.querySelector('.system-status-text').textContent = 'Serveur injoignable';
      });
    }
  }
  refreshHealth();
  setInterval(() => { if (!document.hidden) refreshHealth(); }, 20000);

  window.FG = { esc, icon, toast, fmtTime, fmtDateTime, fmtRelative, isToday, avatar, statusBadge, UNKNOWN, SPOOF };
})();
