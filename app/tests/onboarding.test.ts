import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDatabase, ensureSchema } from "../src/server/db/client";
import { installOrUpdateHaloEngine, runCommand } from "../src/server/halo/engine";
import {
  getHaloEngineSettings,
  saveHaloEngineSettings,
  saveHaloProvider,
} from "../src/server/halo/storage";
import {
  getAppSetting,
  getOnboardingState,
  grandfatherOnboarding,
  markOnboardingComplete,
  resetOnboarding,
  setAppSetting,
} from "../src/server/settings/storage";

function freshDb() {
  const database = createDatabase(":memory:");
  ensureSchema(database.sqlite);
  return database;
}

describe("app settings", () => {
  test("round-trips and overwrites values", () => {
    const { sqlite } = freshDb();
    expect(getAppSetting(sqlite, "missing")).toBeNull();
    setAppSetting(sqlite, "theme", "dark");
    setAppSetting(sqlite, "theme", "light");
    expect(getAppSetting(sqlite, "theme")).toBe("light");
  });
});

describe("onboarding state", () => {
  test("starts incomplete, completes idempotently, resets", () => {
    const { sqlite } = freshDb();
    expect(getOnboardingState(sqlite).completedAt).toBeNull();

    const first = markOnboardingComplete(sqlite);
    expect(first.completedAt).not.toBeNull();
    // Completing again keeps the original timestamp.
    expect(markOnboardingComplete(sqlite).completedAt).toBe(first.completedAt);

    expect(resetOnboarding(sqlite).completedAt).toBeNull();
  });

  test("grandfathers installs that already have a provider", () => {
    const { sqlite } = freshDb();
    grandfatherOnboarding(sqlite);
    expect(getOnboardingState(sqlite).completedAt).toBeNull();

    saveHaloProvider(sqlite, {
      apiKey: "sk-test",
      baseUrl: "https://api.openai.com/v1",
      model: "gpt-4.1-mini",
      name: "OpenAI",
      providerType: "openai",
    });
    grandfatherOnboarding(sqlite);
    expect(getOnboardingState(sqlite).completedAt).not.toBeNull();
  });

  test("grandfathers installs with an installed engine", () => {
    const database = freshDb();
    saveHaloEngineSettings(database.sqlite, {
      commitSha: "abc1234",
      dbPath: database.path,
      status: "installed",
    });
    grandfatherOnboarding(database.sqlite);
    expect(getOnboardingState(database.sqlite).completedAt).not.toBeNull();
  });

  test("leaves genuinely fresh installs incomplete", () => {
    const { sqlite } = freshDb();
    grandfatherOnboarding(sqlite);
    grandfatherOnboarding(sqlite);
    expect(getOnboardingState(sqlite).completedAt).toBeNull();
  });
});

describe("engine install", () => {
  test("stores and reports status detail", () => {
    const database = freshDb();
    saveHaloEngineSettings(database.sqlite, {
      dbPath: database.path,
      status: "installing",
      statusDetail: "Installing Python dependencies…",
    });
    const settings = getHaloEngineSettings(database.sqlite, database.path);
    expect(settings.status).toBe("installing");
    expect(settings.statusDetail).toBe("Installing Python dependencies…");
  });

  test("concurrent installs share one in-flight run", async () => {
    const database = freshDb();
    const installDir = mkdtempSync(join(tmpdir(), "halo-engine-test-"));
    try {
      // A repo URL that fails to clone instantly keeps the test fast while
      // still exercising the real install path.
      saveHaloEngineSettings(database.sqlite, {
        dbPath: database.path,
        installPath: join(installDir, "engine"),
        repoUrl: join(installDir, "missing-repo.git"),
        status: "not_installed",
      });

      const first = installOrUpdateHaloEngine(database);
      const second = installOrUpdateHaloEngine(database);
      expect(second).toBe(first);

      expect(first).rejects.toThrow();
      await first.catch(() => {});

      const settings = getHaloEngineSettings(database.sqlite, database.path);
      expect(settings.status).toBe("error");

      // After the run settles, a new call starts a fresh install.
      const third = installOrUpdateHaloEngine(database);
      expect(third).not.toBe(first);
      await third.catch(() => {});
    } finally {
      rmSync(installDir, { force: true, recursive: true });
    }
  });

  test("missing command errors are actionable", async () => {
    await expect(runCommand(["uv-command-that-does-not-exist-for-halo-test"])).rejects.toThrow(
      "uv-command-that-does-not-exist-for-halo-test was not found",
    );
  });
});
