#!/usr/bin/env bun

import { $ } from "bun";

const email = process.env.GATOR_EMAIL ?? "sam@inference.net";
const apiUrl = process.env.GATOR_API_URL ?? "http://localhost:3000";
const invokeUrl = process.env.GATOR_INVOKE_URL ?? "http://localhost:4300/invoke";
const authToken = process.env.GATOR_AUTH_TOKEN ?? "local-dev-internal-token";

const message =
  process.argv.slice(2).join(" ") ||
  "Query our lists and tell me which contacts in which lists have active todos";

function sqlString(value: string) {
  return value.replaceAll("'", "''");
}

async function queryScalar(sql: string) {
  const output =
    await $`docker exec gator-postgres psql -U gator -d gator -At -c ${sql}`.text();

  const value = output.trim();
  if (!value) {
    throw new Error(`Query returned no rows: ${sql}`);
  }

  return value;
}

async function postJson<T>(url: string, body: unknown, headers: HeadersInit = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...headers,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`);
  }

  return response.json() as Promise<T>;
}

type PrepareResponse = {
  data: {
    agentSessionId: string;
    actorToken: string;
  };
};

async function main() {
  const orgId = await queryScalar(
    `select org_id from "user" where email='${sqlString(email)}' limit 1`,
  );

  const userId = await queryScalar(
    `select id from "user" where email='${sqlString(email)}' limit 1`,
  );

  const prepare = await postJson<PrepareResponse>(
    `${apiUrl}/api/v1/agent-sessions/scheduled/prepare`,
    {
      orgId,
      scheduledJobId: `otel-local-hello-${Date.now()}`,
      allowWrites: false,
      requester: {
        appUserId: userId,
        email,
        role: "admin",
        isAdmin: true,
      },
    },
    {
      authorization: `Bearer ${authToken}`,
    },
  );

  const result = await postJson(invokeUrl, {
    sessionId: prepare.data.agentSessionId,
    message,
    actorToken: prepare.data.actorToken,
  });

  console.log(JSON.stringify(result, null, 2));
}

if (import.meta.main && process.env.NODE_ENV !== "test") {
  await main();
}
