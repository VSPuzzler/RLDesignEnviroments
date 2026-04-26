(function () {
  'use strict';

  const $ = id => document.getElementById(id);

  const GRADE_COLORS = {
    A: 'bg-emerald-500 text-white',
    B: 'bg-green-500 text-white',
    C: 'bg-amber-500 text-white',
    D: 'bg-orange-500 text-white',
    F: 'bg-red-600 text-white',
  };

  const CODE_LINES = [
    { t: '<!DOCTYPE html>',                                           c: '#94a3b8' },
    { t: '<html lang="en">',                                          c: '#86efac' },
    { t: '<head>',                                                    c: '#86efac' },
    { t: '  <meta charset="utf-8">',                                  c: '#94a3b8' },
    { t: '  <style>',                                                 c: '#86efac' },
    { t: '    body  { font-family: system-ui; background:#fff; }',   c: '#7dd3fc' },
    { t: '    nav   { display:flex; padding:0 48px; height:64px; }', c: '#7dd3fc' },
    { t: '    .hero { padding:96px 48px; text-align:center; }',      c: '#7dd3fc' },
    { t: '    h1    { font-size:52px; font-weight:800; }',           c: '#7dd3fc' },
    { t: '    .cta  { background:#4f46e5; color:#fff; }',            c: '#7dd3fc' },
    { t: '    .features { display:grid; grid-template-columns:repeat(3,1fr); }', c: '#7dd3fc' },
    { t: '    .card { border:1px solid #e2e8f0; border-radius:16px; }', c: '#7dd3fc' },
    { t: '  </style>',                                               c: '#86efac' },
    { t: '</head>',                                                   c: '#86efac' },
    { t: '<body>',                                                    c: '#86efac' },
    { t: '  <nav aria-label="Main navigation">',                     c: '#86efac' },
    { t: '    <a href="#" class="logo">Brand</a>',                   c: '#94a3b8' },
    { t: '    <a href="#" class="cta" role="button">Get Started</a>',c: '#94a3b8' },
    { t: '  </nav>',                                                  c: '#86efac' },
    { t: '  <section class="hero" aria-labelledby="h1">',            c: '#86efac' },
    { t: '    <h1 id="h1">Your Headline Here</h1>',                  c: '#fcd34d' },
    { t: '    <p>Supporting description text...</p>',                c: '#94a3b8' },
    { t: '    <a href="#" class="cta" role="button">Get Started Free</a>', c: '#94a3b8' },
    { t: '  </section>',                                             c: '#86efac' },
    { t: '  <section class="features" aria-labelledby="fh">',       c: '#86efac' },
    { t: '    <h2 id="fh">Key features</h2>',                       c: '#fcd34d' },
    { t: '    <article class="card"><h3>Feature One</h3></article>', c: '#94a3b8' },
    { t: '    <article class="card"><h3>Feature Two</h3></article>', c: '#94a3b8' },
    { t: '    <article class="card"><h3>Feature Three</h3></article>',c: '#94a3b8' },
    { t: '  </section>',                                             c: '#86efac' },
    { t: '  <footer><p>© 2025 Brand, Inc.</p></footer>',            c: '#94a3b8' },
    { t: '</body>',                                                   c: '#86efac' },
    { t: '</html>',                                                   c: '#86efac' },
  ];

  const animTimers = {};
  const pendingHtml = {};

  // ── Code animation ──────────────────────────────────────────────────────

  function startCodeAnim(side) {
    const container = $(`codeanim-${side}`);
    const idle = $(`idle-${side}`);
    idle.classList.add('hidden');
    container.classList.remove('hidden');
    container.innerHTML = '';
    let i = 0;
    function tick() {
      if (i >= CODE_LINES.length) return;
      const { t, c } = CODE_LINES[i++];
      const el = document.createElement('div');
      el.textContent = t;
      el.style.color = c;
      container.appendChild(el);
      container.scrollTop = container.scrollHeight;
      animTimers[side] = setTimeout(tick, 90 + Math.random() * 60);
    }
    tick();
  }

  function stopCodeAnim(side) {
    clearTimeout(animTimers[side]);
  }

  // ── Reveal / score ──────────────────────────────────────────────────────

  function revealDesign(side, html, candidateId) {
    stopCodeAnim(side);
    const iframe = $(`iframe-${side}`);
    const loading = $(`loading-${side}`);
    if (candidateId) $(`link-${side}`).href = `/candidate/${candidateId}`;
    iframe.srcdoc = html;
    requestAnimationFrame(() => {
      iframe.style.opacity = '1';
      setTimeout(() => loading.classList.add('hidden'), 500);
    });
  }

  function showScore(side, reward, grade) {
    const colors = GRADE_COLORS[grade] || 'bg-slate-600 text-white';
    const badge = $(`grade-${side}`);
    badge.textContent = grade;
    badge.className = `text-xs font-bold px-2 py-0.5 rounded ${colors}`;
    $(`reward-${side}`).textContent = `${reward} reward`;
    $(`meta-${side}`).classList.remove('hidden');
  }

  function showComparison(winner, pAOverB, rewardA, rewardB) {
    if (winner !== 'tie') {
      $(`winner-${winner}`).classList.remove('hidden');
      $(`box-${winner}`).classList.remove('border-slate-800');
      $(`box-${winner}`).classList.add('border-emerald-600');
    }
    const pct = Math.round(pAOverB * 100);
    const winPct = winner === 'a' ? pct : 100 - pct;
    $('result-text').textContent = winner === 'tie'
      ? `Tie — both designs scored similarly  ·  A: ${rewardA}  ·  B: ${rewardB}`
      : `Design ${winner.toUpperCase()} preferred  ·  ${winPct}% win probability  ·  A: ${rewardA}  ·  B: ${rewardB}`;
    $('result-links').classList.remove('hidden');
    $('result-bar').classList.remove('hidden');
  }

  function resetUI(idleMsg) {
    const msg = idleMsg || 'Enter a prompt above to begin';
    ['a', 'b'].forEach(s => {
      stopCodeAnim(s);
      delete pendingHtml[s];
      $(`meta-${s}`).classList.add('hidden');
      $(`winner-${s}`).classList.add('hidden');
      $(`iframe-${s}`).style.opacity = '0';
      $(`iframe-${s}`).srcdoc = '';
      $(`loading-${s}`).classList.remove('hidden');
      $(`codeanim-${s}`).classList.add('hidden');
      $(`idle-${s}`).classList.remove('hidden');
      $(`idle-${s}`).querySelector('p').textContent = msg;
      $(`box-${s}`).classList.remove('border-emerald-600');
      $(`box-${s}`).classList.add('border-slate-800');
    });
    $('result-bar').classList.add('hidden');
    $('result-links').classList.add('hidden');
    $('status').textContent = '';
  }

  // ── Generate flow ───────────────────────────────────────────────────────

  function dispatch(ev) {
    switch (ev.type) {
      case 'status':
        $('status').textContent = ev.message;
        if (ev.message.includes('Design A')) startCodeAnim('a');
        if (ev.message.includes('Design B')) startCodeAnim('b');
        break;
      case 'html_a':
        pendingHtml['a'] = { html: ev.html, candidateId: ev.candidate_id };
        break;
      case 'score_a':
        if (pendingHtml['a']) revealDesign('a', pendingHtml['a'].html, pendingHtml['a'].candidateId);
        showScore('a', ev.reward, ev.grade);
        break;
      case 'html_b':
        pendingHtml['b'] = { html: ev.html, candidateId: ev.candidate_id };
        break;
      case 'score_b':
        if (pendingHtml['b']) revealDesign('b', pendingHtml['b'].html, pendingHtml['b'].candidateId);
        showScore('b', ev.reward, ev.grade);
        break;
      case 'comparison':
        showComparison(ev.winner, ev.p_a_over_b, ev.reward_a, ev.reward_b);
        break;
      case 'done':
        $('status').textContent = '';
        break;
    }
  }

  async function runGenerate(prompt) {
    const btn = $('gen-btn');
    btn.disabled = true;
    btn.textContent = 'Generating…';
    resetUI('Generating…');

    try {
      const resp = await fetch('/api/generate-pair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      if (!resp.ok) { $('status').textContent = `Error ${resp.status}: ${resp.statusText}`; return; }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch { continue; }
          dispatch(ev);
        }
      }
    } catch (err) {
      $('status').textContent = 'Error: ' + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate';
    }
  }

  // ── Demo flow ───────────────────────────────────────────────────────────

  async function loadDemo() {
    $('status').textContent = 'Loading demo…';
    startCodeAnim('a');
    startCodeAnim('b');

    try {
      const data = await fetch('/api/demo').then(r => r.json());

      // Reveal A first, then B with a short stagger for visual effect
      setTimeout(() => {
        revealDesign('a', data.html_a, null);
        showScore('a', data.reward_a, data.grade_a);
      }, 600);

      setTimeout(() => {
        revealDesign('b', data.html_b, null);
        showScore('b', data.reward_b, data.grade_b);
      }, 1100);

      setTimeout(() => {
        showComparison(data.winner, data.p_a_over_b, data.reward_a, data.reward_b);
        $('status').textContent = '';
      }, 1400);
    } catch (err) {
      $('status').textContent = 'Error loading demo: ' + err.message;
    }
  }

  // ── Tabs ────────────────────────────────────────────────────────────────

  function activateTab(tab) {
    const isDemo = tab === 'demo';

    // Tab button styles
    $('tab-generate').className = isDemo
      ? 'px-5 py-2 rounded-lg text-sm font-semibold text-slate-400 hover:text-white transition-colors'
      : 'px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white transition-colors';
    $('tab-demo').className = isDemo
      ? 'px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white transition-colors'
      : 'px-5 py-2 rounded-lg text-sm font-semibold text-slate-400 hover:text-white transition-colors';

    // Panel visibility
    $('panel-generate').classList.toggle('hidden', isDemo);
    $('panel-demo').classList.toggle('hidden', !isDemo);

    // Reset boxes then load appropriate content
    if (isDemo) {
      resetUI('Loading demo…');
      loadDemo();
    } else {
      resetUI('Enter a prompt above to begin');
    }
  }

  // ── Boot ────────────────────────────────────────────────────────────────

  document.addEventListener('DOMContentLoaded', () => {
    $('tab-generate').addEventListener('click', () => activateTab('generate'));
    $('tab-demo').addEventListener('click', () => activateTab('demo'));

    $('gen-btn').addEventListener('click', () => {
      const val = $('prompt').value.trim();
      if (!val) { $('prompt').focus(); return; }
      runGenerate(val);
    });

    $('prompt').addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('gen-btn').click();
    });
  });
})();
