import { fileURLToPath } from "node:url";

const projectRoot = fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
const viewUrl = Bun.env.HALO_VIEW_URL ?? "http://127.0.0.1:5173";
const dbPath = Bun.env.HALO_DB_PATH ?? `${projectRoot}/data/halo-canvas.sqlite`;

const env: Record<string, string> = {};
for (const [key, value] of Object.entries(Bun.env)) {
  if (value !== undefined) {
    env[key] = value;
  }
}
env.HALO_DB_PATH = dbPath;
env.HALO_PROJECT_ROOT = projectRoot;
env.HALO_RUNNER_PATH = `${projectRoot}/scripts/halo-local-runner.py`;
env.HALO_VIEW_URL = viewUrl;

const dbPush = Bun.spawnSync([process.execPath, "run", "db:push"], {
  cwd: projectRoot,
  env,
  stderr: "inherit",
  stdout: "inherit",
});

if (dbPush.exitCode !== 0) {
  process.exit(dbPush.exitCode ?? 1);
}

const web = Bun.spawn([process.execPath, "run", "dev:web"], {
  cwd: projectRoot,
  env,
  stderr: "inherit",
  stdout: "inherit",
});

try {
  await waitForView(viewUrl);
} catch (error) {
  web.kill();
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}

const desktop = Bun.spawn([process.execPath, "x", "electrobun", "dev", "--watch"], {
  cwd: projectRoot,
  env,
  stderr: "inherit",
  stdout: "inherit",
});

let shuttingDown = false;

async function shutdown(exitCode = 0) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  desktop.kill();
  web.kill();
  await Promise.allSettled([desktop.exited, web.exited]);
  process.exit(exitCode);
}

process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());

web.exited.then((exitCode) => {
  if (!shuttingDown) {
    console.error(`Vite dev server exited with code ${exitCode}.`);
    void shutdown(exitCode);
  }
});

const exitCode = await desktop.exited;
await shutdown(exitCode);

async function waitForView(url: string) {
  const deadline = Date.now() + 20_000;
  let lastError: unknown;

  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(1_000),
      });
      if (response.status < 500) {
        return;
      }
      lastError = new Error(`Vite responded with ${response.status}`);
    } catch (error) {
      lastError = error;
    }

    await Bun.sleep(250);
  }

  throw new Error(
    `Timed out waiting for the Vite dev server at ${url}.${
      lastError instanceof Error ? ` Last error: ${lastError.message}` : ""
    }`,
  );
}
