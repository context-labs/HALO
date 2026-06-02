import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { tmpdir } from "node:os";

const TWEMOJI_SOURCE_URL =
  "https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f607.svg";
const ICONSET_DIR = "icon.iconset";
const PREVIEW_ICON = "assets/app-icon.png";
const SOURCE_SVG = "assets/app-icon.svg";
const NOTICE_FILE = "assets/app-icon.NOTICE.md";

const ICONSET_FILES = [
  ["icon_16x16.png", 16],
  ["icon_16x16@2x.png", 32],
  ["icon_32x32.png", 32],
  ["icon_32x32@2x.png", 64],
  ["icon_128x128.png", 128],
  ["icon_128x128@2x.png", 256],
  ["icon_256x256.png", 256],
  ["icon_256x256@2x.png", 512],
  ["icon_512x512.png", 512],
  ["icon_512x512@2x.png", 1024],
] as const;

mkdirSync(ICONSET_DIR, { recursive: true });
mkdirSync(dirname(PREVIEW_ICON), { recursive: true });

const svg = await fetchTwemojiSvg();
writeFileSync(SOURCE_SVG, svg);
writeFileSync(NOTICE_FILE, noticeText());

for (const [fileName, size] of ICONSET_FILES) {
  renderSvgToPng(size, `${ICONSET_DIR}/${fileName}`);
}

copyFileSync(`${ICONSET_DIR}/icon_512x512@2x.png`, PREVIEW_ICON);

console.log(`Fetched Twemoji source from ${TWEMOJI_SOURCE_URL}`);
console.log(`Generated ${ICONSET_FILES.length} macOS iconset PNGs`);
console.log(`Generated ${PREVIEW_ICON}`);
console.log(`Generated ${SOURCE_SVG}`);
console.log(`Generated ${NOTICE_FILE}`);

async function fetchTwemojiSvg() {
  const response = await fetch(TWEMOJI_SOURCE_URL);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch Twemoji SVG: ${response.status} ${response.statusText}`,
    );
  }
  return response.text();
}

function renderSvgToPng(size: number, outputPath: string) {
  const tempDir = mkdtempSync(join(tmpdir(), "halo-canvas-icon-"));
  try {
    const result = Bun.spawnSync([
      "qlmanage",
      "-t",
      "-s",
      String(size),
      "-o",
      tempDir,
      SOURCE_SVG,
    ]);

    if (result.exitCode !== 0) {
      throw new Error(
        `qlmanage failed while rendering ${outputPath}\n${result.stderr.toString()}`,
      );
    }

    const thumbnailPath = join(tempDir, `${basename(SOURCE_SVG)}.png`);
    if (!existsSync(thumbnailPath)) {
      throw new Error(`qlmanage did not produce ${thumbnailPath}`);
    }

    copyFileSync(thumbnailPath, outputPath);
  } finally {
    rmSync(tempDir, { force: true, recursive: true });
  }
}

function noticeText() {
  return `# App Icon Attribution

The HALO app icon uses the Twemoji "Smiling Face with Halo" graphic
(Unicode U+1F607).

- Source: ${TWEMOJI_SOURCE_URL}
- Twemoji repository: https://github.com/twitter/twemoji
- Graphics license: CC-BY 4.0
`;
}
