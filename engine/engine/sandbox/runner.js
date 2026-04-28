// HALO sandbox runner: Deno + Pyodide WASM.
//
// Adapted from DSPy's primitives/runner.js — trimmed to the minimum HALO
// needs: no tools, no SUBMIT, no env vars, no host writes, no network.
//
// Wire protocol: JSON-RPC 2.0 over stdin/stdout, one message per line.
//
// Methods (host → runner):
//   mount_file   {host_path, virtual_path}            read host file, write to pyodide FS
//   bootstrap    {trace_path, index_path}             load trace_store, build user globals
//   execute      {code}                               run user Python; return {exit_code, stdout, stderr}
//   shutdown                                          notification, exits the loop
//
// All embedded Python lives in sibling ``pyodide_runtime.py`` (capture +
// exec helpers) and ``pyodide_trace_compat.py`` (stdlib trace store).
// runner.js reads both at startup and runs them inside Pyodide; per-call
// requests just invoke the resulting ``halo_bootstrap`` /
// ``halo_execute`` Python functions over JSON-RPC.
//
// Permissions are hardcoded by the parent (``--allow-read`` covering the
// runner script, its sibling .py files, the Deno cache, and the per-run
// trace + index). We never request --allow-net, --allow-write,
// --allow-env, --allow-run.

// Version pin must match ``_PYODIDE_VERSION`` in ``pyodide_client.py``:
// the client looks up cached wheels and the npm package directory by that
// exact version string. Without the pin Deno would resolve to whatever is
// latest on npm, populate a different cache directory, and the client's
// existence check would silently fail — leaving ``run_code`` quietly
// disabled the next time pyodide ships a release.
import pyodideModule from "npm:pyodide@0.29.3/pyodide.js";

// =============================================================================
// JSON-RPC helpers
// =============================================================================

const JSONRPC_PROTOCOL_ERRORS = {
  ParseError: -32700,
  InvalidRequest: -32600,
  MethodNotFound: -32601,
};

const JSONRPC_APP_ERRORS = {
  RuntimeError: -32007,
  SandboxError: -32008,
  Unknown: -32099,
};

const jsonrpcResult = (result, id) =>
  JSON.stringify({ jsonrpc: "2.0", result, id });

const jsonrpcError = (code, message, id, data = null) => {
  const err = { code, message };
  if (data) err.data = data;
  return JSON.stringify({ jsonrpc: "2.0", error: err, id });
};

// Surface unhandled rejections as JSON-RPC errors so the host parser does not
// see a Deno crash splatted across stdout/stderr.
globalThis.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  console.log(jsonrpcError(
    JSONRPC_APP_ERRORS.RuntimeError,
    `Unhandled async error: ${event.reason?.message || event.reason}`,
    null,
  ));
});

// =============================================================================
// Pyodide bootstrap
// =============================================================================

const pyodide = await pyodideModule.loadPyodide();

// Resolve sibling files via ``import.meta.url``. ``Deno.readTextFileSync``
// and ``Deno.readDirSync`` are gated by ``--allow-read``, which the parent
// process scopes to exactly these paths (see ``Sandbox._build_argv``).
const runtimePath = new URL("./pyodide_runtime.py", import.meta.url).pathname;
const engineInitPath = new URL("../__init__.py", import.meta.url).pathname;
const tracesPkgPath = new URL("../traces", import.meta.url).pathname;

// Stage the host's ``engine`` package into Pyodide's WASM filesystem at
// ``/halo/engine/`` so the in-Pyodide bootstrap can simply
// ``import engine.traces.trace_store``. The runtime adds ``/halo`` to
// ``sys.path`` once its module loads. Only ``engine/__init__.py`` plus
// the ``engine/traces/`` subtree are staged — that's the entire import
// graph the WASM-side TraceStore needs, and there's no point copying
// host-only modules (sandbox itself, agents, tools, etc.).
pyodide.FS.mkdirTree("/halo/engine");
pyodide.FS.writeFile(
  "/halo/engine/__init__.py",
  Deno.readTextFileSync(engineInitPath),
);
copyDirToPyodide(tracesPkgPath, "/halo/engine/traces");

// Define ``halo_bootstrap`` and ``halo_execute`` in the Pyodide globals.
// Single ``runPython`` at boot — every per-call request from the host
// just invokes the live functions, no Python codegen at request time.
pyodide.runPython(Deno.readTextFileSync(runtimePath));

function copyDirToPyodide(srcAbs, destVirtual) {
  // Recursively copy ``.py`` files under ``srcAbs`` into the Pyodide
  // virtual FS at ``destVirtual``. Skips ``__pycache__`` because nothing
  // inside it is import-relevant and the bytecode would be host-arch.
  pyodide.FS.mkdirTree(destVirtual);
  for (const entry of Deno.readDirSync(srcAbs)) {
    if (entry.name === "__pycache__") continue;
    const srcChild = `${srcAbs}/${entry.name}`;
    const destChild = `${destVirtual}/${entry.name}`;
    if (entry.isFile && entry.name.endsWith(".py")) {
      pyodide.FS.writeFile(destChild, Deno.readTextFileSync(srcChild));
    } else if (entry.isDirectory) {
      copyDirToPyodide(srcChild, destChild);
    }
  }
}

// =============================================================================
// Method handlers
// =============================================================================

function mountFile(params) {
  const hostPath = params.host_path;
  const virtualPath = params.virtual_path;
  if (!hostPath || !virtualPath) {
    throw new Error("mount_file requires host_path and virtual_path");
  }
  // Deno.readFileSync requires --allow-read covering hostPath. The parent
  // process scopes --allow-read to exactly the trace + index files.
  const contents = Deno.readFileSync(hostPath);
  ensurePyodideDir(virtualPath);
  pyodide.FS.writeFile(virtualPath, contents);
  return { mounted: virtualPath };
}

function ensurePyodideDir(virtualPath) {
  const segments = virtualPath.split("/").slice(1, -1);
  let cur = "";
  for (const seg of segments) {
    cur += "/" + seg;
    try {
      pyodide.FS.stat(cur);
    } catch {
      pyodide.FS.mkdir(cur);
    }
  }
}

function callPyResult(fn, ...args) {
  // The Python helpers return plain dicts; convert to a JS object so we
  // can JSON-stringify directly. ``Object.fromEntries`` keeps numeric
  // ``exit_code`` numeric rather than coercing to a wrapped PyProxy.
  const result = fn(...args);
  return result.toJs({ dict_converter: Object.fromEntries });
}

function bootstrap(params) {
  const tracePath = params.trace_path;
  const indexPath = params.index_path;
  if (!tracePath || !indexPath) {
    throw new Error("bootstrap requires trace_path and index_path");
  }
  return callPyResult(pyodide.globals.get("halo_bootstrap"), tracePath, indexPath);
}

function executeCode(params) {
  return callPyResult(pyodide.globals.get("halo_execute"), params.code || "");
}

// =============================================================================
// Main loop
// =============================================================================

// Preload numpy + pandas so user code can rely on them without a per-execute
// load delay. These are the only data libraries HALO supports.
// Preload numpy + pandas (user analysis surface) and pydantic (which
// the staged ``engine.traces`` package imports). Pyodide resolves
// transitive dependencies from its lockfile, so listing ``pydantic`` is
// enough to pull pydantic_core + typing_extensions + annotated_types +
// typing_inspection. All required wheels are pre-cached by
// ``Sandbox._ensure_pyodide_wheels``; this call is offline.
await pyodide.loadPackage(["numpy", "pandas", "pydantic"]);

// Tell the host we're ready. The host parses the first line as a sentinel.
console.log(jsonrpcResult({ ready: true }, 0));

// ``stream: true`` makes the decoder buffer trailing partial UTF-8
// sequences across chunks. Without it, a multi-byte character split
// across two stdin reads (entirely possible for any non-ASCII content
// in user code or trace data) would emit U+FFFD on each side of the
// split and corrupt the JSON-RPC message.
const decoder = new TextDecoder("utf-8");
let buffer = "";

for await (const chunk of Deno.stdin.readable) {
  buffer += decoder.decode(chunk, { stream: true });
  let newlineIdx;
  while ((newlineIdx = buffer.indexOf("\n")) !== -1) {
    const line = buffer.slice(0, newlineIdx);
    buffer = buffer.slice(newlineIdx + 1);
    if (!line.trim()) continue;

    let input;
    try {
      input = JSON.parse(line);
    } catch (err) {
      console.log(jsonrpcError(
        JSONRPC_PROTOCOL_ERRORS.ParseError,
        `Invalid JSON input: ${err.message}`,
        null,
      ));
      continue;
    }

    if (typeof input !== "object" || input === null || input.jsonrpc !== "2.0") {
      console.log(jsonrpcError(
        JSONRPC_PROTOCOL_ERRORS.InvalidRequest,
        "Invalid Request: not a JSON-RPC 2.0 message",
        null,
      ));
      continue;
    }

    const method = input.method;
    const params = input.params || {};
    const requestId = input.id;

    if (method === "shutdown") {
      Deno.exit(0);
    }

    try {
      let result;
      if (method === "mount_file") {
        result = mountFile(params);
      } else if (method === "bootstrap") {
        result = bootstrap(params);
      } else if (method === "execute") {
        result = executeCode(params);
      } else {
        console.log(jsonrpcError(
          JSONRPC_PROTOCOL_ERRORS.MethodNotFound,
          `Method not found: ${method}`,
          requestId,
        ));
        continue;
      }
      console.log(jsonrpcResult(result, requestId));
    } catch (err) {
      console.log(jsonrpcError(
        JSONRPC_APP_ERRORS.SandboxError,
        err?.message || String(err),
        requestId,
      ));
    }
  }
}
