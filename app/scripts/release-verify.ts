import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";

const args = parseArgs(Bun.argv.slice(2));
const artifactDir = args["artifacts-dir"] ?? "artifacts";
const channel = args.channel ?? process.env.HALO_RELEASE_CHANNEL ?? "stable";
const appFileName = channel === "stable" ? "HALO" : `HALO-${channel}`;

if (!existsSync(artifactDir)) {
  fail(`Artifact directory does not exist: ${artifactDir}`);
}

const files = (await readdir(artifactDir)).sort();
const hasMacArtifacts = files.some((file) => file.startsWith(`${channel}-macos-arm64-`));
const hasLinuxArtifacts = files.some((file) => file.startsWith(`${channel}-linux-x64-`));

if (!hasMacArtifacts && !hasLinuxArtifacts) {
  fail(`No ${channel} macOS arm64 or Linux x64 artifacts found in ${artifactDir}`);
}

if (hasMacArtifacts) {
  expectFile(`${channel}-macos-arm64-${appFileName}.dmg`);
  expectFile(`${channel}-macos-arm64-${appFileName}.app.tar.zst`);
  expectFile(`${channel}-macos-arm64-update.json`);
}

if (hasLinuxArtifacts) {
  expectFile(`${channel}-linux-x64-${appFileName}-Setup.tar.gz`);
  expectFile(`${channel}-linux-x64-${appFileName}.tar.zst`);
  expectFile(`${channel}-linux-x64-update.json`);
}

expectFile("manifest.json");
expectFile("SHA256SUMS");

await verifyChecksums();
await verifyManifest();

console.log(`Release artifacts verified in ${artifactDir}`);

function expectFile(fileName: string) {
  if (!existsSync(join(artifactDir, fileName))) {
    fail(`Missing expected artifact: ${fileName}`);
  }
}

async function verifyChecksums() {
  const checksumPath = join(artifactDir, "SHA256SUMS");
  const lines = (await readFile(checksumPath, "utf8"))
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    fail("SHA256SUMS is empty");
  }

  for (const line of lines) {
    const match = line.match(/^([a-f0-9]{64})\s+(.+)$/i);
    if (!match) {
      fail(`Malformed checksum line: ${line}`);
    }
    const expected = match[1]!.toLowerCase();
    const fileName = match[2]!;
    const filePath = join(artifactDir, fileName);
    if (!existsSync(filePath)) {
      fail(`Checksum references missing file: ${fileName}`);
    }
    const actual = await sha256File(filePath);
    if (actual !== expected) {
      fail(`Checksum mismatch for ${fileName}: expected ${expected}, got ${actual}`);
    }
  }
}

async function verifyManifest() {
  const manifest = (await Bun.file(join(artifactDir, "manifest.json")).json()) as {
    app?: string;
    channel?: string;
    checksums?: { fileName?: string; url?: string; signatureFileName?: string; signatureUrl?: string };
    artifacts?: Array<{ fileName?: string; sha256?: string; url?: string }>;
  };

  if (manifest.app !== "HALO") {
    fail("manifest.json app must be HALO");
  }
  if (manifest.channel !== channel) {
    fail(`manifest.json channel must be ${channel}`);
  }
  if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length === 0) {
    fail("manifest.json must contain artifacts");
  }
  if (
    manifest.checksums?.fileName !== "SHA256SUMS" ||
    manifest.checksums.signatureFileName !== "SHA256SUMS.sigstore.json" ||
    !manifest.checksums.url ||
    !manifest.checksums.signatureUrl
  ) {
    fail("manifest.json must describe SHA256SUMS and its sigstore bundle URL");
  }

  for (const artifact of manifest.artifacts) {
    if (!artifact.fileName || !artifact.sha256 || !artifact.url) {
      fail("manifest artifact entries must include fileName, sha256, and url");
    }
    if (!existsSync(join(artifactDir, artifact.fileName))) {
      fail(`manifest references missing file: ${artifact.fileName}`);
    }
  }
}

async function sha256File(filePath: string) {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(await Bun.file(filePath).arrayBuffer());
  return hasher.digest("hex");
}

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

function fail(message: string): never {
  console.error(message);
  process.exit(1);
}
