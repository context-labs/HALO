import type { Database } from "bun:sqlite";

const ONBOARDING_COMPLETED_AT = "onboarding_completed_at";

export type OnboardingState = {
  completedAt: string | null;
};

export function getAppSetting(sqlite: Database, key: string): string | null {
  const row = sqlite
    .query<{ value: string }, [string]>(
      `SELECT value
       FROM app_settings
       WHERE key = ?
       LIMIT 1`,
    )
    .get(key);
  return row?.value ?? null;
}

export function setAppSetting(sqlite: Database, key: string, value: string) {
  const now = Date.now();
  sqlite
    .query(
      `INSERT INTO app_settings (key, value, updated_at)
       VALUES (?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET
         value = excluded.value,
         updated_at = excluded.updated_at`,
    )
    .run(key, value, now);
}

export function deleteAppSetting(sqlite: Database, key: string) {
  sqlite.query("DELETE FROM app_settings WHERE key = ?").run(key);
}

export function getOnboardingState(sqlite: Database): OnboardingState {
  return { completedAt: getAppSetting(sqlite, ONBOARDING_COMPLETED_AT) };
}

export function markOnboardingComplete(sqlite: Database): OnboardingState {
  if (!getAppSetting(sqlite, ONBOARDING_COMPLETED_AT)) {
    setAppSetting(sqlite, ONBOARDING_COMPLETED_AT, new Date().toISOString());
  }
  return getOnboardingState(sqlite);
}

export function resetOnboarding(sqlite: Database): OnboardingState {
  deleteAppSetting(sqlite, ONBOARDING_COMPLETED_AT);
  return getOnboardingState(sqlite);
}

/**
 * Existing installs predate onboarding — anyone who already set up the
 * engine or a model provider has effectively completed it, so don't show
 * the welcome flow on their next launch. Runs once per boot.
 */
export function grandfatherOnboarding(sqlite: Database) {
  if (getAppSetting(sqlite, ONBOARDING_COMPLETED_AT)) return;
  const engineConfigured =
    (sqlite
      .query<{ n: number }, []>(
        `SELECT COUNT(*) AS n
         FROM halo_engine_settings
         WHERE status = 'installed' OR commit_sha IS NOT NULL`,
      )
      .get()?.n ?? 0) > 0;
  const hasProviders =
    (sqlite
      .query<{ n: number }, []>("SELECT COUNT(*) AS n FROM halo_model_providers")
      .get()?.n ?? 0) > 0;
  if (engineConfigured || hasProviders) {
    markOnboardingComplete(sqlite);
  }
}
