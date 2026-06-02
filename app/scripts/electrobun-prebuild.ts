if (Bun.env.HALO_VIEW_URL) {
  console.log(`Skipping web asset build because HALO_VIEW_URL=${Bun.env.HALO_VIEW_URL}`);
  process.exit(0);
}

const result = Bun.spawnSync([process.execPath, "run", "build:web"], {
  cwd: new URL("..", import.meta.url).pathname,
  stderr: "inherit",
  stdout: "inherit",
});

if (result.exitCode !== 0) {
  process.exit(result.exitCode ?? 1);
}
