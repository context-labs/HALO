import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { Screen, type BrowserWindow } from "electrobun/bun";

type WindowFrame = {
  height: number;
  width: number;
  x: number;
  y: number;
};

const MIN_HEIGHT = 520;
const MIN_WIDTH = 760;

export function loadWindowFrame(
  appDataDir: string,
  fallback: WindowFrame,
): WindowFrame {
  const path = windowStatePath(appDataDir);
  if (!existsSync(path)) return centeredFallback(fallback);

  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    const frame = normalizeFrame(parsed, fallback);
    return isVisibleOnAnyDisplay(frame) ? frame : centeredFallback(fallback);
  } catch {
    return centeredFallback(fallback);
  }
}

export function persistWindowFrame(
  appDataDir: string,
  window: BrowserWindow,
) {
  const path = windowStatePath(appDataDir);
  const save = () => {
    try {
      const frame = normalizeFrame(window.getFrame(), window.getFrame());
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, JSON.stringify(frame, null, 2));
    } catch {
      // This is a convenience feature; failures should not affect app shutdown.
    }
  };

  const interval = setInterval(save, 2_000);
  return {
    save,
    stop() {
      clearInterval(interval);
      save();
    },
  };
}

function windowStatePath(appDataDir: string) {
  return join(appDataDir, "window-state.json");
}

function normalizeFrame(value: unknown, fallback: WindowFrame): WindowFrame {
  if (!value || typeof value !== "object") return fallback;
  const maybe = value as Partial<WindowFrame>;
  return {
    height: clampNumber(maybe.height, fallback.height, MIN_HEIGHT, 10_000),
    width: clampNumber(maybe.width, fallback.width, MIN_WIDTH, 10_000),
    x: finiteNumber(maybe.x, fallback.x),
    y: finiteNumber(maybe.y, fallback.y),
  };
}

function centeredFallback(fallback: WindowFrame): WindowFrame {
  const display = Screen.getPrimaryDisplay();
  const workArea = display.workArea;
  if (!workArea.width || !workArea.height) return fallback;

  return {
    ...fallback,
    x: Math.round(workArea.x + Math.max(0, (workArea.width - fallback.width) / 2)),
    y: Math.round(workArea.y + Math.max(0, (workArea.height - fallback.height) / 2)),
  };
}

function isVisibleOnAnyDisplay(frame: WindowFrame) {
  const displays = Screen.getAllDisplays();
  if (displays.length === 0) return true;

  return displays.some((display) => {
    const area = display.workArea;
    const intersectsHorizontally =
      frame.x + Math.min(frame.width, 80) > area.x &&
      frame.x < area.x + area.width - 40;
    const intersectsVertically =
      frame.y + Math.min(frame.height, 80) > area.y &&
      frame.y < area.y + area.height - 40;
    return intersectsHorizontally && intersectsVertically;
  });
}

function clampNumber(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
) {
  const finite = finiteNumber(value, fallback);
  return Math.max(min, Math.min(max, finite));
}

function finiteNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
