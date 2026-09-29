// people.js — recherche, dates lisibles et retrait d'une personne (boîte de confirmation).
(() => {
  'use strict';
  const { toast, fmtRelative } = window.FG;
  const $ = id => document.getElementById(id);

  document.querySelectorAll('[data-last-seen]').forEach(el => {
    if (el.dataset.lastSeen) el.textContent = `Dernier passage ${fmtRelative(el.dataset.lastSeen)}`;
  });

  const search = $('people-search');
  function applySearch() {
    const q = (search?.value || '').trim().toLowerCase();
    let visible = 0;
    document.querySelectorAll('.person-card').forEach(c => {
      const show = !q || c.dataset.name.includes(q);
      c.hidden = !show;
      if (show) visible++;
    });
    const noResult = $('people-noresult');
    if (noResult) noResult.hidden = visible > 0 || !q;
  }
  search?.addEventListener('input', applySearch);

  const dialog = $('delete-dialog');
  let target = null;
  document.addEventListener('click', e => {
    const btn = e.target.closest('[data-delete]');
    if (!btn) return;
    target = btn;
    $('delete-name').textContent = btn.dataset.delete;
    dialog.showModal();
  });

  dialog.addEventListener('close', async () => {
    if (dialog.returnValue !== 'confirm' || !target) return;
    const name = target.dataset.delete;
    const card = target.closest('.person-card');
    target.disabled = true;
    try {
      const r = await fetch(`/people/${encodeURIComponent(name)}/delete`, { method: 'POST', headers: { Accept: 'application/json' } });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `Erreur serveur (${r.status})`);
      card.classList.add('removing');
      setTimeout(() => {
        card.remove();
        const left = document.querySelectorAll('.person-card').length;
        $('people-count').textContent = left;
        if (!left) { $('people-empty').hidden = false; $('people-grid')?.remove(); }
      }, 300);
      toast(data.message || `${name} a été retiré(e).`, 'success');
    } catch (err) {
      target.disabled = false;
      toast(`Impossible de retirer ${name} : ${err.message}`, 'error');
    }
  });
})();
