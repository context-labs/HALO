export interface Env {
  HALO_RELEASES: R2Bucket;
}

const CONTENT_TYPES: Record<string, string> = {
  ".dmg": "application/x-apple-diskimage",
  ".gz": "application/gzip",
  ".json": "application/json; charset=utf-8",
  ".sh": "text/x-shellscript; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".zst": "application/zstd",
};

export default {
  async fetch(request: Request, env: Env) {
    const key = objectKeyForRequest(request);
    if (!key) {
      return new Response("Not found", { status: 404 });
    }

    const object = await env.HALO_RELEASES.get(key);
    if (!object) {
      return new Response("Not found", { status: 404 });
    }

    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set("etag", object.httpEtag);
    headers.set("cache-control", cacheControlForKey(key));
    if (!headers.has("content-type")) {
      headers.set("content-type", contentTypeForKey(key));
    }

    if (request.method === "HEAD") {
      return new Response(null, { headers });
    }

    return new Response(object.body, { headers });
  },
};

function objectKeyForRequest(request: Request) {
  const url = new URL(request.url);
  const pathname = url.pathname.replace(/\/+$/, "");

  if (pathname === "/halo/install.sh") {
    return "halo/install.sh";
  }

  if (pathname.startsWith("/halo/releases/")) {
    return pathname.slice(1);
  }

  return null;
}

function cacheControlForKey(key: string) {
  if (key.endsWith("/manifest.json") || key.endsWith("/SHA256SUMS")) {
    return "public, max-age=300";
  }
  if (key === "halo/install.sh" || key.endsWith("-update.json")) {
    return "public, max-age=300";
  }
  return "public, max-age=31536000, immutable";
}

function contentTypeForKey(key: string) {
  for (const [extension, contentType] of Object.entries(CONTENT_TYPES)) {
    if (key.endsWith(extension)) {
      return contentType;
    }
  }
  return "application/octet-stream";
}
