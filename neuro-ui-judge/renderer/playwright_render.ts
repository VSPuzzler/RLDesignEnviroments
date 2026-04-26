/**
 * TypeScript renderer (parallel to playwright_render.py).
 *
 * This file is provided per spec for a future Next.js-side renderer. The
 * primary MVP path uses the Python renderer to keep the stack single-language.
 *
 * Usage (after `npm i playwright`):
 *   ts-node renderer/playwright_render.ts <candidate_id> <html_path> <out_dir>
 */
import { chromium } from "playwright";
import * as fs from "fs";
import * as path from "path";

const EXTRACT_JS = `
() => {
  const INTERACTIVE_TAGS = new Set(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA']);
  const CTA_KEYWORDS = ['sign up', 'get started', 'try', 'buy', 'subscribe',
                        'start', 'download', 'join', 'continue', 'next',
                        'submit', 'create', 'go'];
  function visible(el) {
    const r = el.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1) return false;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none') return false;
    if (parseFloat(s.opacity) < 0.05) return false;
    return true;
  }
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    if (!visible(el)) continue;
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    const text = (el.innerText || el.textContent || '').slice(0, 200).trim();
    const interactive = INTERACTIVE_TAGS.has(el.tagName)
                        || el.getAttribute('role') === 'button';
    if (!text && !interactive && !['IMG','SVG','CANVAS','VIDEO'].includes(el.tagName))
      continue;
    out.push({
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role'),
      text: text || null,
      bbox: { x: r.x, y: r.y, width: r.width, height: r.height },
      font_size_px: parseFloat(s.fontSize) || null,
      color: s.color,
      background_color: s.backgroundColor,
      is_interactive: interactive,
    });
  }
  return { elements: out, visible_text: document.body.innerText || '' };
}
`;

export async function render(
  candidateId: string,
  htmlPath: string,
  outDir: string,
  viewport = { width: 1440, height: 900 },
) {
  fs.mkdirSync(outDir, { recursive: true });
  const screenshotPath = path.join(outDir, `${candidateId}.png`);
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport });
  const page = await ctx.newPage();
  await page.goto(`file://${path.resolve(htmlPath)}`, { waitUntil: "networkidle" });
  await page.screenshot({ path: screenshotPath });
  const extracted = await page.evaluate(EXTRACT_JS);
  await browser.close();
  return { candidateId, screenshotPath, ...extracted };
}

if (require.main === module) {
  const [id, htmlPath, outDir] = process.argv.slice(2);
  render(id, htmlPath, outDir).then((r) => console.log(JSON.stringify(r)));
}
