// HALO sandbox runner: Deno + Pyodide WASM.
//
// Adapted from DSPy's primitives/runner.js — trimmed to the minimum HALO
// needs: no tools, no SUBMIT, no env vars, no host writes, no network.
//
// Wire protocol: JSON-RPC 2.0 over stdin/stdout, one message per line.
//
// Methods (host → runner):
//   mount_file   {host_path, virtual_path}   read host file, write to pyodide FS
//   inject_text  {virtual_path, text}        write inline text into pyodide FS
//   bootstrap    {code}                      run setup Python once; build globals
//   execute      {code}                      run user Python; return {exit_code, stdout, stderr}
//   shutdown                                 notification, exits the loop
//
// Permissions are hardcoded by the parent (--allow-read=<runner>,<deno cache>,<trace>,<index>).
// We never request --allow-net, --allow-write, --allow-env, --allow-run.

import pyodideModule from "npm:pyodide/pyodide.js";

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

// Initialize the host module that holds the per-run globals dict. The
// bootstrap method populates `_halo_runtime.user_globals` once with
// trace_store + numpy/pandas; execute reuses it across calls.
pyodide.runPython(`
import sys, types
_halo_runtime = types.ModuleType("_halo_runtime")
_halo_runtime.user_globals = None
sys.modules["_halo_runtime"] = _halo_runtime
`);

// Per-execute Python wrapper: redirects stdout/stderr to StringIO buffers so
// we can capture both, plus surface the traceback in stderr on failure.
const PYTHON_RUN_TEMPLATE = `
import sys, io, traceback
import _halo_runtime

_halo_buf_stdout = io.StringIO()
_halo_buf_stderr = io.StringIO()
_halo_old_stdout, _halo_old_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = _halo_buf_stdout, _halo_buf_stderr

_halo_exit_code = 0
try:
    if _halo_runtime.user_globals is None:
        raise RuntimeError("sandbox not bootstrapped: bootstrap() must be called before execute()")
    code = pyodide_user_code  # set via pyodide.globals.set
    exec(compile(code, "<sandbox>", "exec"), _halo_runtime.user_globals, _halo_runtime.user_globals)
except BaseException:
    _halo_exit_code = 1
    traceback.print_exc()
finally:
    sys.stdout, sys.stderr = _halo_old_stdout, _halo_old_stderr

_halo_result = {
    "exit_code": _halo_exit_code,
    "stdout": _halo_buf_stdout.getvalue(),
    "stderr": _halo_buf_stderr.getvalue(),
}
`;

// Bootstrap script wrapper: runs once after mounts to populate the user
// globals dict. Same stdout/stderr capture + traceback so a failed bootstrap
// produces an actionable error to the host.
const PYTHON_BOOTSTRAP_TEMPLATE = `
import sys, io, traceback
import _halo_runtime

_halo_buf_stdout = io.StringIO()
_halo_buf_stderr = io.StringIO()
_halo_old_stdout, _halo_old_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = _halo_buf_stdout, _halo_buf_stderr

_halo_exit_code = 0
try:
    code = pyodide_bootstrap_code
    bootstrap_globals = {"__name__": "__halo_bootstrap__"}
    exec(compile(code, "<bootstrap>", "exec"), bootstrap_globals, bootstrap_globals)
    if "user_globals" not in bootstrap_globals:
        raise RuntimeError("bootstrap script must define a 'user_globals' dict")
    _halo_runtime.user_globals = bootstrap_globals["user_globals"]
except BaseException:
    _halo_exit_code = 1
    traceback.print_exc()
finally:
    sys.stdout, sys.stderr = _halo_old_stdout, _halo_old_stderr

_halo_result = {
    "exit_code": _halo_exit_code,
    "stdout": _halo_buf_stdout.getvalue(),
    "stderr": _halo_buf_stderr.getvalue(),
}
`;

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

function injectText(params) {
  const virtualPath = params.virtual_path;
  const text = params.text;
  if (!virtualPath || typeof text !== "string") {
    throw new Error("inject_text requires virtual_path and text");
  }
  ensurePyodideDir(virtualPath);
  pyodide.FS.writeFile(virtualPath, new TextEncoder().encode(text));
  return { injected: virtualPath };
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

async function bootstrap(params) {
  const code = params.code || "";
  pyodide.globals.set("pyodide_bootstrap_code", code);
  await pyodide.runPythonAsync(PYTHON_BOOTSTRAP_TEMPLATE);
  const result = pyodide.globals.get("_halo_result").toJs({ dict_converter: Object.fromEntries });
  return result;
}

async function executeCode(params) {
  const code = params.code || "";
  pyodide.globals.set("pyodide_user_code", code);
  await pyodide.runPythonAsync(PYTHON_RUN_TEMPLATE);
  const result = pyodide.globals.get("_halo_result").toJs({ dict_converter: Object.fromEntries });
  return result;
}

// =============================================================================
// Main loop
// =============================================================================

// Preload numpy + pandas so user code can rely on them without a per-execute
// load delay. These are the only data libraries HALO supports.
await pyodide.loadPackage(["numpy", "pandas"]);

// Tell the host we're ready. The host parses the first line as a sentinel.
console.log(jsonrpcResult({ ready: true }, 0));

const decoder = new TextDecoder();
let buffer = "";

for await (const chunk of Deno.stdin.readable) {
  buffer += decoder.decode(chunk);
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
      } else if (method === "inject_text") {
        result = injectText(params);
      } else if (method === "bootstrap") {
        result = await bootstrap(params);
      } else if (method === "execute") {
        result = await executeCode(params);
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
