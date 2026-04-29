<!-- <p align="center">
  <a href="https://github.com/context-labs/uwu">
    <img src="https://em-content.zobj.net/thumbs/240/apple/354/smiling-face-with-halo_1f607.png" alt="😇" width="100" height="100" style="vertical-align:middle;"></span>
  </a>
  <br>
  <h1>HALO</h1>
</p> -->

<h1 align="center">
  <br>
  <a href="https://github.com/context-labs/uwu"><img src="https://em-content.zobj.net/thumbs/240/apple/354/smiling-face-with-halo_1f607.png" alt="😇" width="150" style="border-radius:8px;"></a>
   <br>
  HALO
  <br>
</h1>

<h4 align="center">✨ RLM-based Automatic Agent Optimization Loop ✨</h4>

<p align="center">
  <a href="https://x.com/inference_net">
    <img alt="X (formerly Twitter)" src="https://img.shields.io/badge/X-@inference.net-1DA1F2?style=flat&logo=x&logoColor=white" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://github.com/context-labs/halo">
    <img alt="GitHub" src="https://img.shields.io/github/stars/context-labs/halo?style=social" />
  </a>

</p>

<p align="center">
  <a href="#what-is-this">What is this?</a> •
  <a href="#install">Install</a> •
  <a href="#why-an-rlm">Why RLM?</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#development">Development</a> •
  <a href="#contributing">Contributing</a>
</p>

## What is this?

HALO (Hierarchical Agent Loop Optimization) is a methodology for building recursively self-improving agent harnesses using [RLMs](https://github.com/alexzhang13/rlm). This repository contains:
- Information on HALO methodology.
- A Python package that implements the core HALO-RLM engine. [View on PyPI](https://pypi.org/project/halo-engine/)
- A demo project that shows how to build HALO loops for your agents using the Python package. [View demo](/demo/openai-agents-sdk-demo/)
- Benchmarking examples applying HALO to popular agent benchmarks. (View [AppWorld](#appworld)).

## HALO Loop

The core HALO loop is suprisingly simple:

1. Collect execution traces from your agent harness. HALO uses OpenTelemetry-compatible tracing.
2. Feed traces into HALO-RLM engine.
3. The engine decomposes the traces to understand common failure modes and across harness executions and produces a report with it’s findings.
4. This report is fed into a coding agent like Cursor or Claude Code to generate and apply a set of changes to your harness.
5. The harness is then re-deployed, more traces are gathered, and the cycle repeats.

HALO is great at finding issues in production agent deployments. We find high-traffic environments tend to generate more data with higher variance across executions, creating the type of issues that HALO is great at identifying.

### Why an RLM?

A general-purpose harness like Claude Code is the wrong tool for trace analysis. This isn’t because the model isn’t smart, but because traces can get extremely long, and you need a specialized toolkit in order to make observations about systemic agentic behavior. We noticed in our testing that harnesses like CC would often overfit to an error present in a single/few traces rather than generalize to harness-level problems. This led us to creating a specialized form of a RLM.

<img src="./assets//halo-rlm.png" alt="rlm"  style="border-radius:8px;" width="600">

## Get Started

### Install

Install the HALO engine + CLI from PyPI:

```bash
pip install halo-engine

# Verify installation
halo --help
```

### Usage

1. [Integrate Tracing](docs/integrations/openai-agents-sdk.md)
2. Collect traces by running your agent
3. Run the HALO engine, see the [CLI](/halo_cli/README.md) docs for more info
```bash
export OPENAI_API_KEY=...
halo path_to_your_traces.jsonl
```

We have provided a [simple demo](/demo/openai-agents-sdk-demo/) and an [AppWorld](#appworld) demo.

## Benchmarks

HALO is consistently capable of driving improvements on benchmarks, solely by optimizing the harness.

### AppWorld

We applied HALO to the [AppWorld](https://appworld.dev/) benchmark, a set of agentic tasks that assess the LLM’s ability to use multi-app services like Spotify, Venmo, file systems, and phone contacts. We tested HALO’s ability to improve harnesses for both Gemini 3 Flash and Sonnet 4.6. We iterated on the harness using the `dev` split, and then used the `test_normal` split as a proxy to verify that improvements did not come from overfitting.

The feedback from HALO Engine surfaced failures in the harnesses such as hallucinated tool calls, redundant arguments in tools, refusal loops, and semantic correctness issues. Each issue mapped cleanly to a direct prompt edit. HALO’s claims were independently verified from the source trace files with the findings holding up under scrutiny.

<img src="./assets//halo-app-world-sgc.png" alt="app-world-sgc"  style="border-radius:8px;">
<!-- 
  Note: Table cell styling is still limited in GitHub Markdown rendering,
  and border-radius is not supported, but background color and padding usually work.
  If this does not display as desired, you will need to update the image asset itself
  to include padding and a black background.
-->
The peak improvements over baseline were substantial for both models. For Gemini 3 Flash, dev SGC went from 36.8% to 52.6% (+15.8 points) and test_normal SGC went from 37.5% to 48.2% (+10.7 points). For Sonnet 4.6, dev SGC went from 73.7% to 89.5% (+15.8 points) and test_normal SGC went from 62.5% to 73.2% (+10.7 points).

## Development

Local development against this repo uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`go-task`](https://taskfile.dev/) as the task runner.

### Setup

```bash
git clone https://github.com/context-labs/HALO
cd HALO
task env:setup
```

`task env:setup` installs `uv` (if missing), syncs the venv from `uv.lock`, and configures the repo's git hooks. After that, the `halo` CLI is available via `uv run halo ...` (or activate `.venv/`).

### Common tasks

Run `task --list` for the full list. The ones you'll use most:

| Task                    | What it does                                                                    |
| ----------------------- | ------------------------------------------------------------------------------- |
| `task check`            | Run all pre-commit checks: pinned-versions, lint, format, typecheck, unit tests |
| `task check:fix`        | Same, but auto-fix lint/format issues                                           |
| `task test:unit`        | Unit tests under `tests/unit/`                                                  |
| `task test:integration` | Integration tests under `tests/integration/`                                    |

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a pull request.
