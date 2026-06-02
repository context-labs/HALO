import type { ElectrobunConfig } from "electrobun";
import { readFileSync } from "node:fs";

const packageJson = JSON.parse(
  readFileSync(new URL("./package.json", import.meta.url), "utf8"),
) as { version: string };

const useRemoteDevView = Boolean(process.env.HALO_VIEW_URL);
const releaseChannel =
  process.env.HALO_RELEASE_CHANNEL ?? process.env.ELECTROBUN_BUILD_ENV ?? "stable";
const releaseBaseRoot = (
  process.env.HALO_RELEASE_BASE_URL ?? "https://inference.net/halo/releases"
).replace(/\/+$/, "");
const shouldSignMac = process.env.HALO_SKIP_CODESIGN !== "1";
const shouldNotarizeMac = shouldSignMac && process.env.HALO_SKIP_NOTARIZE !== "1";

export default {
  app: {
    name: "HALO",
    identifier: "net.inference.halo",
    version: packageJson.version,
    description: "A local desktop OpenTelemetry trace monitor for AI agent development.",
  },
  runtime: {
    exitOnLastWindowClosed: true,
  },
  scripts: {
    preBuild: "scripts/electrobun-prebuild.ts",
  },
  build: {
    targets: process.env.HALO_ELECTROBUN_TARGETS ?? "current",
    bun: {
      entrypoint: "src/bun/index.ts",
      sourcemap: "inline",
    },
    mac: {
      codesign: shouldSignMac,
      notarize: shouldNotarizeMac,
      createDmg: true,
      icons: "icon.iconset",
    },
    win: {
      icon: "assets/app-icon.png",
    },
    linux: {
      bundleCEF: true,
      defaultRenderer: "cef",
      icon: "assets/app-icon.png",
    },
    copy: {
      ...(useRemoteDevView ? {} : { "dist/client": "views/mainview" }),
      "scripts/halo-local-runner.py": "scripts/halo-local-runner.py",
    },
    watch: useRemoteDevView
      ? ["src/server"]
      : ["src/mainview", "src/routes", "src/server", "src/router.tsx"],
    watchIgnore: ["data/**"],
  },
  release: {
    baseUrl: `${releaseBaseRoot}/${releaseChannel}`,
  },
} satisfies ElectrobunConfig;
