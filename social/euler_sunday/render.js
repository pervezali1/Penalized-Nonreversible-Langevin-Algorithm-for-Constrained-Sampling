// Render build/html/*.html to images/*.jpg at exactly 1080 x 1350.
//   node render.js [name-filter]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const here = __dirname;
  const manifest = JSON.parse(fs.readFileSync(path.join(here, 'build', 'manifest.json')));
  const filter = process.argv[2] || '';
  fs.mkdirSync(path.join(here, 'images'), { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 1 });
  for (const s of manifest) {
    if (filter && !s.name.startsWith(filter)) continue;
    await page.goto('file://' + path.join(here, 'build', 'html', s.name + '.html'));
    await page.evaluate(() => document.fonts.ready);
    // overflow check: the frame's content must fit the canvas
    const over = await page.evaluate(() => {
      const f = document.querySelector('.frame');
      const st = document.querySelector('.stage');
      return { frame: f.scrollHeight - f.clientHeight, stage: st ? st.scrollHeight - st.clientHeight : 0 };
    });
    if (over.frame > 0 || over.stage > 0) console.log('OVERFLOW', s.name, JSON.stringify(over));
    await page.screenshot({ path: path.join(here, 'images', s.name + '.jpg'), type: 'jpeg', quality: 92 });
  }
  await browser.close();
  console.log('done');
})();
