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

  // Fake code lines shown while the backend renders + scores.
  const CODE_LINES = [
    { t: '<!DOCTYPE html>',                            c: '#94a3b8' },
    { t: '<html lang="en">',                           c: '#86efac' },
    { t: '<head>',                                     c: '#86efac' },
    { t: '  <meta charset="utf-8">',                   c: '#94a3b8' },
    { t: '  <style>',                                  c: '#86efac' },
    { t: '    body  { font-family: system-ui; }',      c: '#7dd3fc' },
    { t: '    nav   { display: flex; padding: 16px; }',c: '#7dd3fc' },
    { t: '    .hero { padding: 80px; text-align: center; }', c: '#7dd3fc' },
    { t: '    h1    { font-size: 48px; font-weight: 800; }', c: '#7dd3fc' },
    { t: '    .cta  { background: #6366f1; color: #fff; }',  c: '#7dd3fc' },
    { t: '    .features { display: grid; grid-template-columns: repeat(3,1fr); }', c: '#7dd3fc' },
    { t: '  </style>',                                 c: '#86efac' },
    { t: '</head>',                                    c: '#86efac' },
    { t: '<body>',                                     c: '#86efac' },
    { t: '  <nav>',                                    c: '#86efac' },
    { t: '    <div class="logo">Brand</div>',          c: '#94a3b8' },
    { t: '    <a href="#" class="cta">Get Started</a>',c: '#94a3b8' },
    { t: '  </nav>',                                   c: '#86efac' },
    { t: '  <section class="hero">',                   c: '#86efac' },
    { t: '    <h1>Your Headline Here</h1>',             c: '#fcd34d' },
    { t: '    <p>Supporting description text...</p>',  c: '#94a3b8' },
    { t: '    <a class="cta" role="button">Get Started Free</a>', c: '#94a3b8' },
    { t: '  </section>',                               c: '#86efac' },
    { t: '  <section class="features">',               c: '#86efac' },
    { t: '    <div class="card"><h3>Feature One</h3></div>', c: '#94a3b8' },
    { t: '    <div class="card"><h3>Feature Two</h3></div>', c: '#94a3b8' },
    { t: '    <div class="card"><h3>Feature Three</h3></div>', c: '#94a3b8' },
    { t: '  </section>',                               c: '#86efac' },
    { t: '  <footer>© 2025 Brand</footer>',            c: '#94a3b8' },
    { t: '</body>',                                    c: '#86efac' },
    { t: '</html>',                                    c: '#86efac' },
  ];

  const animTimers = {};
  const pendingHtml = {};

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
      animTimers[side] = setTimeout(tick, 100 + Math.random() * 60);
    }
    tick();
  }

  function stopCodeAnim(side) {
    clearTimeout(animTimers[side]);
  }

  function revealDesign(side) {
    const pending = pendingHtml[side];
    if (!pending) return;
    const { html, candidateId } = pending;

    stopCodeAnim(side);

    const iframe = $(`iframe-${side}`);
    const loading = $(`loading-${side}`);
    $(`link-${side}`).href = `/candidate/${candidateId}`;

    iframe.srcdoc = html;
    // fade in the iframe over the code animation
    requestAnimationFrame(() => {
      iframe.style.opacity = '1';
      // hide loading overlay after transition
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

  function showComparison(ev) {
    if (ev.winner !== 'tie') {
      $(`winner-${ev.winner}`).classList.remove('hidden');
      $(`box-${ev.winner}`).classList.remove('border-slate-800');
      $(`box-${ev.winner}`).classList.add('border-emerald-600');
    }
    const pct = Math.round(ev.p_a_over_b * 100);
    const winPct = ev.winner === 'a' ? pct : 100 - pct;
    $('result-text').textContent = ev.winner === 'tie'
      ? `Tie — both designs scored similarly  ·  A: ${ev.reward_a}  ·  B: ${ev.reward_b}`
      : `Design ${ev.winner.toUpperCase()} preferred  ·  ${winPct}% win probability  ·  A: ${ev.reward_a}  ·  B: ${ev.reward_b}`;
    $('result-links').classList.remove('hidden');
    $('result-bar').classList.remove('hidden');
  }

  function resetUI() {
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
      $(`box-${s}`).classList.remove('border-emerald-600');
      $(`box-${s}`).classList.add('border-slate-800');
    });
    $('result-bar').classList.add('hidden');
    $('result-links').classList.add('hidden');
    $('status').textContent = '';
  }

  function dispatch(ev) {
    const status = $('status');
    switch (ev.type) {
      case 'status':
        status.textContent = ev.message;
        if (ev.message.includes('Design A')) startCodeAnim('a');
        if (ev.message.includes('Design B')) startCodeAnim('b');
        break;
      case 'html_a':
        pendingHtml['a'] = { html: ev.html, candidateId: ev.candidate_id };
        break;
      case 'score_a':
        revealDesign('a');
        showScore('a', ev.reward, ev.grade);
        break;
      case 'html_b':
        pendingHtml['b'] = { html: ev.html, candidateId: ev.candidate_id };
        break;
      case 'score_b':
        revealDesign('b');
        showScore('b', ev.reward, ev.grade);
        break;
      case 'comparison':
        showComparison(ev);
        break;
      case 'done':
        status.textContent = '';
        break;
    }
  }

  async function runGenerate(prompt) {
    const btn = $('gen-btn');
    btn.disabled = true;
    btn.textContent = 'Generating…';
    resetUI();

    try {
      const resp = await fetch('/api/generate-pair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (!resp.ok) {
        $('status').textContent = `Error ${resp.status}: ${resp.statusText}`;
        return;
      }

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

  document.addEventListener('DOMContentLoaded', () => {
    const btn = $('gen-btn');
    const promptEl = $('prompt');

    btn.addEventListener('click', () => {
      const val = promptEl.value.trim();
      if (!val) { promptEl.focus(); return; }
      runGenerate(val);
    });

    promptEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) btn.click();
    });
  });
})();
