import { readdir } from "node:fs/promises";
import { basename, join } from "node:path";

type ArtifactKind =
  | "macos-dmg"
  | "linux-installer"
  | "update-archive"
  | "update-info"
  | "patch"
  | "unknown";

type ReleaseArtifact = {
  fileName: string;
  url: string;
  sha256: string;
  size: number;
  channel: string;
  platform: string;
  arch: string;
  kind: ArtifactKind;
};

type Manifest = {
  app: "HALO";
  version: string;
  channel: string;
  generatedAt: string;
  baseUrl: string;
  checksums: {
    fileName: "SHA256SUMS";
    url: string;
    signatureFileName: "SHA256SUMS.sigstore.json";
    signatureUrl: string;
  };
  artifacts: ReleaseArtifact[];
};

const args = parseArgs(Bun.argv.slice(2));
const artifactDir = args["artifacts-dir"] ?? "artifacts";
const channel = args.channel ?? process.env.HALO_RELEASE_CHANNEL ?? "stable";
const baseUrl = (
  args["base-url"] ??
  process.env.HALO_RELEASE_ARTIFACT_BASE_URL ??
  `https://inference.net/halo/releases/${channel}`
).replace(/\/+$/, "");
const packageJson = (await Bun.file("package.json").json()) as { version: string };

const fileNames = (await readdir(artifactDir))
  .filter((fileName) => !fileName.startsWith("."))
  .filter((fileName) => !isGeneratedReleaseFile(fileName))
  .sort();

if (fileNames.length === 0) {
  fail(`No release artifacts found in ${artifactDir}`);
}

const artifacts: ReleaseArtifact[] = [];
const checksumLines: string[] = [];

for (const fileName of fileNames) {
  const filePath = join(artifactDir, fileName);
  const file = Bun.file(filePath);
  const sha256 = await sha256File(filePath);
  const parsed = parseArtifactName(fileName, channel);
  artifacts.push({
    fileName,
    url: `${baseUrl}/${encodeURIComponent(fileName)}`,
    sha256,
    size: file.size,
    channel: parsed.channel,
    platform: parsed.platform,
    arch: parsed.arch,
    kind: parsed.kind,
  });
  checksumLines.push(`${sha256}  ${fileName}`);
}

const manifest: Manifest = {
  app: "HALO",
  version: packageJson.version,
  channel,
  generatedAt: new Date().toISOString(),
  baseUrl,
  checksums: {
    fileName: "SHA256SUMS",
    url: `${baseUrl}/SHA256SUMS`,
    signatureFileName: "SHA256SUMS.sigstore.json",
    signatureUrl: `${baseUrl}/SHA256SUMS.sigstore.json`,
  },
  artifacts,
};

await Bun.write(join(artifactDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
await Bun.write(join(artifactDir, "SHA256SUMS"), `${checksumLines.join("\n")}\n`);

console.log(`Wrote ${join(artifactDir, "manifest.json")}`);
console.log(`Wrote ${join(artifactDir, "SHA256SUMS")}`);

function parseArgs(argv: string[]) {
  const parsed: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg?.startsWith("--")) continue;
    const [key, inlineValue] = arg.slice(2).split("=", 2);
    if (!key) continue;
    if (inlineValue !== undefined) {
      parsed[key] = inlineValue;
      continue;
    }
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      parsed[key] = next;
      index += 1;
    } else {
      parsed[key] = "true";
    }
  }
  return parsed;
}

function isGeneratedReleaseFile(fileName: string) {
  return (
    fileName === "manifest.json" ||
    fileName === "SHA256SUMS" ||
    fileName.endsWith(".sigstore.json") ||
    fileName.endsWith(".sig") ||
    fileName.endsWith(".minisig")
  );
}

function parseArtifactName(fileName: string, fallbackChannel: string) {
  const match = fileName.match(/^(stable|canary)-([^-]+)-([^-]+)-(.+)$/);
  const parsedChannel = match?.[1] ?? fallbackChannel;
  const platform = match?.[2] ?? "unknown";
  const arch = match?.[3] ?? "unknown";
  const rest = match?.[4] ?? basename(fileName);

  return {
    channel: parsedChannel,
    platform,
    arch,
    kind: artifactKind(rest),
  };
}

function artifactKind(fileName: string): ArtifactKind {
  if (fileName.endsWith(".dmg")) return "macos-dmg";
  if (fileName.endsWith("-Setup.tar.gz")) return "linux-installer";
  if (fileName.endsWith(".tar.zst")) return "update-archive";
  if (fileName === "update.json" || fileName.endsWith("-update.json")) return "update-info";
  if (fileName.endsWith(".patch")) return "patch";
  return "unknown";
}

async function sha256File(filePath: string) {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(await Bun.file(filePath).arrayBuffer());
  return hasher.digest("hex");
}

function fail(message: string): never {
  console.error(message);
  process.exit(1);
}
