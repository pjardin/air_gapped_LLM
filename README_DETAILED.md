# air_gapped_LLM — detailed walkthrough

> This is the **detailed** README — explanations of why each step exists, tradeoffs, platform caveats, recovery instructions when things go wrong, and the reasoning behind every config value. For a terse copy-paste checklist with no commentary, see **[README.md](./README.md)**.

A fully offline system for indexing a code repository and answering questions about it later. No network access at training or inference time. **Currently in design phase — no implementation yet.**

## Planned architecture

- **Inference runtime:** [Ollama](https://ollama.com) (wraps llama.cpp), GGUF quantized models, CPU-only.
- **Base model:** Llama-3.2-1B-Instruct as the **current POC default** — the whole walkthrough is set up for it end-to-end (YAML, Modelfile, eval). The Step 4a model menu also covers Llama-3.2-3B for the production scale-up after the POC succeeds, and Llama-3.1-8B / Llama-3.3-70B for production runs on a Linux target.
- **Domain adaptation:** LoRA continued pretraining on the repo's source code. **This is the only knowledge source** — no retrieval at query time. All repo knowledge lives in the model's weights.
- **Target hardware:** mid-tier CPU server, 16–64 cores, AVX-512, no GPU.

Ollama handles **inference only**. Fine-tuning uses a separate Python toolchain (HuggingFace `transformers` + `peft` on CPU); the resulting LoRA adapter is converted to GGUF for Ollama to serve.

**Known tradeoffs of the fine-tune-only approach:** the model can only answer about code that was in the last training run (no freshness), can't cite source files, and is prone to hallucinating specific code details that weren't memorized strongly during training. Strongest use case is *generating code in the repo's style*; weakest is *answering precise questions about specific functions*. Re-training is required after meaningful repo changes.

## Repo layout

Concrete scaffolding lives alongside the README, ready to copy onto the air-gapped target:

```
.
├── README.md
├── modelfiles/
│   # (no Modelfile for the Step 2 serving model — `llama3.1:8b` is pulled directly
│   #  from Ollama's library and used as-is; library models already have proper templates)
│   └── Modelfile.llama-3.2-1b-cisc187        # Step 4g — fine-tune tag (FROM swaps: base 1B → merged GGUF)
├── configs/
│   ├── goose-config.yaml                  # Step 3 — copy to ~/.config/goose/config.yaml
│   ├── cisc187_pt.yaml                    # Step 4e — LLaMA-Factory training config
│   └── dataset_info_patch.json            # Step 4d — entry to merge into LLaMA-Factory/data/dataset_info.json
├── scripts/
│   ├── prepare_cisc187.py                 # Step 4d — repo → CPT JSONL
│   └── run_train.py                       # Step 4f — wrapper that patches torch.mps then calls llamafactory.cli.main()
└── repo_to_fine_tune/
    └── cisc187-reader-master/             # Source corpus for the POC fine-tune
```

Paths inside the Modelfiles and configs use placeholders like `/REPLACE/WITH/PATH/TO/...` — update them for your air-gapped layout before running `ollama create` or `llamafactory-cli train`.

### Where to put things on disk

Every command below assumes two folders sit directly under your home directory on the target machine:

```
~/air_gapped_llm/      # this repo (clone or extract here)
~/llf-bundle/          # the LLaMA-Factory bundle that Step 4a assembles and Step 4b extracts into $HOME
```

The README, the YAML in `configs/cisc187_pt.yaml`, the Modelfiles, and `scripts/run_train.py` all assume this layout. The dataset path inside `dataset_info_patch.json`, the `dataset_dir` field in the YAML, the merged-model paths in Step 4g — every one of them is written as `~/llf-bundle/...` or `~/air_gapped_llm/...`. Put both anywhere else and you'll spend the next hour grep-replacing paths.

Concretely on macOS or Linux:

```bash
# Place the repo
cd ~
git clone <wherever-this-repo-lives> air_gapped_llm
# or unzip a transferred copy into ~/air_gapped_llm

# The llf-bundle directory gets populated during Step 4a/4b — just make sure
# you extract the bundle tarball into $HOME so it lands at ~/llf-bundle.
```

---

## Step 0 — Install Python 3.11 (Anaconda)

The pipeline standardizes on **Python 3.11** across both the networked machine and the air-gapped target. 3.11 was chosen because it has:

- Wheels available for every package in this pipeline (LLaMA-Factory, recent PyTorch ≥ 2.4, transformers, etc.)
- ~10–15% faster interpreter performance than 3.10 (PEP 659 specialization)
- Long support window (security fixes through October 2027)
- Mature ABI — older C-extension packages don't break the way they sometimes do on 3.12+

We install it via **Anaconda**, which bundles Python 3.11 plus ~250 preinstalled data-science packages (NumPy, pandas, SciPy, scikit-learn, Jupyter, etc.) in a single offline installer. That matters for the air-gapped target — you can't `pip install` extras over the network there, so getting the scientific stack pre-bundled saves trouble.

**Pin to Anaconda 2023.09-0.** It's the last Anaconda release that bundles Python 3.11 by default — 2024.x and later switched to 3.12.

> **Licensing note:** Anaconda is free for individual use, academic/research use, and organizations under 200 employees. Larger commercial orgs need a paid license under Anaconda's 2024 terms change — confirm with your org before deploying.

### Download (on the networked machine)

Anaconda publishes two installer formats for macOS: `.pkg` (GUI / `installer(8)` driven) and `.sh` (shell script). **Use the `.sh` even on macOS** — the `.pkg` writes to `/anaconda3` at the root of the boot volume, which on modern macOS (Catalina+) fails with `"The package is attempting to install content to the system volume."` because the system volume is read-only. The `.sh` installs into `$HOME/anaconda3` and doesn't need sudo, matching the Linux pattern exactly.

```bash
# macOS Apple Silicon
curl -L https://repo.anaconda.com/archive/Anaconda3-2023.09-0-MacOSX-arm64.sh   -o Anaconda3-2023.09-0-MacOSX-arm64.sh

# macOS Intel
curl -L https://repo.anaconda.com/archive/Anaconda3-2023.09-0-MacOSX-x86_64.sh  -o Anaconda3-2023.09-0-MacOSX-x86_64.sh

# Windows x86_64
curl -L https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Windows-x86_64.exe -o Anaconda3-2023.09-0-Windows-x86_64.exe

# Linux x86_64
curl -L https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh   -o Anaconda3-2023.09-0-Linux-x86_64.sh

# Linux ARM64 (Graviton, Pi, etc.)
curl -L https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-aarch64.sh  -o Anaconda3-2023.09-0-Linux-aarch64.sh
```

### Install on the air-gapped target

**macOS:**

```bash
# Use the .sh installer (no sudo, no system-volume conflict)
bash Anaconda3-2023.09-0-MacOSX-arm64.sh -b -p $HOME/anaconda3     # or -x86_64.sh on Intel
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
python3.11 --version
```

**Linux:**

```bash
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p $HOME/anaconda3
# -b = batch mode (no prompts), -p = install prefix
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
python3.11 --version
```

The flags `-b -p $HOME/anaconda3` are identical between macOS and Linux — `-b` is batch mode (accept all defaults, no interactive prompts), `-p` is the install prefix.

**Windows:** double-click the `.exe`, follow the GUI installer, check "Add Anaconda3 to my PATH environment variable" when prompted. Verify in a new Command Prompt with `python --version`.

### Verify

```bash
python3.11 --version              # → Python 3.11.x
python3.11 -m pip --version       # → pip X.Y.Z from .../python3.11/site-packages/pip
```

Once both commands work, proceed to Step 1.

---

## Step 1 — Download Ollama (do this on a networked machine, then transfer)

Pick the package that matches the target machine. Run these on a box with internet access, then move the file to the air-gapped host.

```bash
# macOS (app bundle, requires macOS 14+)
curl -L https://ollama.com/download/Ollama.dmg -o Ollama.dmg

# Windows (installer)
curl -L https://ollama.com/download/OllamaSetup.exe -o OllamaSetup.exe

# Linux x86_64 (Intel/AMD CPU)
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst

# Linux ARM64 (Raspberry Pi, Graviton, ARM servers)
curl -L https://ollama.com/download/ollama-linux-arm64.tar.zst -o ollama-linux-arm64.tar.zst

# Linux x86_64 + AMD GPU (ROCm add-on, extract on top of amd64)
curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst -o ollama-linux-amd64-rocm.tar.zst
```

### Install on Linux after transfer

The Linux archives are `.tar.zst`. Extract into a prefix on `PATH` (e.g. `/usr`):

```bash
sudo tar -I zstd -xvf ollama-linux-amd64.tar.zst -C /usr
# Optional ROCm add-on (same prefix, layered on top of amd64)
sudo tar -I zstd -xvf ollama-linux-amd64-rocm.tar.zst -C /usr
```

### Install on macOS / Windows after transfer

- macOS: open `Ollama.dmg`, drag `Ollama.app` to `/Applications`, launch once to register the CLI.
- Windows: run `OllamaSetup.exe`.

### Configure Ollama environment variables (on the target machine, before first run)

> **Do this before your first `ollama serve`, not after.** A running Ollama daemon captures its environment at launch and won't pick up later `.bashrc` edits — you'd have to stop and restart it (see *If Ollama is already running* further down). The default `OLLAMA_KEEP_ALIVE=5m` is the painful one on a CPU box: every 5 minutes of idle, the next request pays a 10–30 second reload tax while the multi-GB GGUF gets re-mmapped. Setting it *before* the first model loads avoids ever experiencing it.

Append the following to the shell rc file on the air-gapped box (`~/.bashrc` on Linux, `~/.zshrc` on macOS's default shell). These tune Ollama for the long-running, CPU-only, air-gapped use case — without them, Ollama runs with defaults aimed at a multi-GPU desktop chatbot setup.

```bash
# ~/.bashrc  (Linux)   or   ~/.zshrc  (macOS default shell)

# Keep loaded models resident in RAM indefinitely. The default unloads after
# 5 minutes of inactivity, which causes a painful cold start every time —
# reloading a multi-GB GGUF from disk on CPU takes 10–30+ seconds. Set to
# "-1" to never unload, or a duration like "24h" / "1h" for a finite window.
export OLLAMA_KEEP_ALIVE=-1

# Don't auto-prune unused model blobs on startup. We're air-gapped: if
# Ollama silently deletes a blob, we can't re-download it.
export OLLAMA_NOPRUNE=true

# Only keep one model loaded in RAM at a time. Mid-tier CPU box, limited
# RAM — a second resident model thrashes both. Raise if you have RAM
# headroom and want to A/B sizes without reload latency.
export OLLAMA_MAX_LOADED_MODELS=1

# Serialize requests rather than parallelizing them. CPU inference is
# bottlenecked on memory bandwidth; letting one request use all cores
# beats two requests fighting for them.
export OLLAMA_NUM_PARALLEL=1

# Default bind is 127.0.0.1:11434 (local only). Uncomment to expose on
# the LAN — only if your air-gap actually has a LAN you trust.
# export OLLAMA_HOST=0.0.0.0:11434

# Optional: relocate the model store if $HOME is on a small partition.
# Default is ~/.ollama/models.
# export OLLAMA_MODELS=/data/ollama/models
```

Reload with `source ~/.bashrc` (or `source ~/.zshrc`), or just open a new terminal.

**Heads-up — `.bashrc` doesn't always reach Ollama.** It depends on how the daemon is launched:

- **Linux, `ollama serve` started from a shell** → reads `.bashrc`. Works as expected.
- **Linux, systemd service** (some packaged installs) → does **not** read `.bashrc`. Either disable the service and run `ollama serve` manually, or set the vars in the unit: `sudo systemctl edit ollama`, add one `Environment="OLLAMA_KEEP_ALIVE=-1"` line per variable under `[Service]`, then `sudo systemctl restart ollama`.
- **macOS, Ollama.app GUI** → launched by `launchd`, ignores `.zshrc`. Either quit the app and run `ollama serve` from a terminal that has the vars set, or use `launchctl setenv OLLAMA_KEEP_ALIVE -1` (per variable; survives until reboot — wrap in a LaunchAgent for persistence).
- **Windows** → set via System Properties → Environment Variables, or PowerShell: `[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "-1", "User")`.

### Start the daemon

Once env vars are in place:

```bash
# Foreground — good for first-run, logs print to the terminal
ollama serve

# Detached background
nohup ollama serve > ~/ollama.log 2>&1 &
```

If you installed via the macOS app or a systemd service, the daemon may already be running — see *Is the daemon running?* below. Verify it's up:

```bash
ollama --version
ollama ps     # lists currently-loaded models; empty list = daemon up, idle
```

### If Ollama is already running — restart it to pick up env-var changes

A running daemon froze its environment at launch; later edits to `.bashrc` don't reach the running process. Sourcing `.bashrc` in your shell only updates *your shell* — Ollama is a separate process with its own (now stale) env. To apply changes, stop Ollama and restart it **from a shell that has the new env loaded**. The how depends on how Ollama is currently running:

**Foreground `ollama serve` in a terminal:**

```bash
# In the running terminal: Ctrl-C to stop it.
# In a fresh terminal (which picks up the new .bashrc automatically):
ollama serve
```

**Background `ollama serve` (`nohup … &` or similar):**

```bash
pkill ollama
source ~/.bashrc                                  # or just open a new shell
nohup ollama serve > ~/ollama.log 2>&1 &
```

**Linux systemd service:**

Systemd ignores `.bashrc` entirely — sourcing won't help. Set the env vars in the unit instead (as in the caveats above), then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**macOS Ollama.app:**

Quit the app from the menu-bar icon (the GUI server doesn't read `.zshrc`), then either:

```bash
# Option 1: run from a terminal that has the env loaded
source ~/.zshrc && ollama serve

# Option 2: push vars into launchd, then relaunch Ollama.app
launchctl setenv OLLAMA_KEEP_ALIVE -1
launchctl setenv OLLAMA_NOPRUNE true
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
launchctl setenv OLLAMA_NUM_PARALLEL 1
# Then open /Applications/Ollama.app again.
```

**Windows:**

Stop Ollama via Task Manager (or `Stop-Service Ollama` in an admin PowerShell), set the env vars under *System Properties → Environment Variables*, then start Ollama from the Start Menu.

**Confirm changes actually took effect:**

```bash
ollama ps
# After loading a model, the UNTIL column should read "Forever"
# (proves OLLAMA_KEEP_ALIVE=-1 reached the daemon).
```

---

## Step 2 — Pull the base model from Ollama's library

> **Purpose of this step:** put a model into Ollama so it can be used for both interactive chat AND as the brain for the Step 3 Goose agent. The model must be **tools-capable** so Goose can give it real filesystem tools.
>
> **Training uses a different file.** Fine-tuning needs the HuggingFace-format model (FP16 safetensors + tokenizer files) that is downloaded separately in **Step 4a** via `hf download`. By the time you start training, you will have *both* the Ollama-managed serving model from this step AND a separate HuggingFace-format download for LLaMA-Factory.

> **Why pull from Ollama's library instead of a custom GGUF?** Ollama only marks a model as tools-capable if its chat template advertises tool-call placeholders explicitly. Library models like `llama3.1:8b` have this baked in; custom-imported GGUFs from Unsloth often don't, and Goose will reject them with `<tag> does not support tools`. The earlier draft of this project used a custom GGUF + `Modelfile` flow — it hit exactly that error. The fix is to use Ollama's library version, which "just works" with Goose out of the box.

### Pull the default model

```bash
ollama pull llama3.1:8b
ollama show llama3.1:8b | grep -i tools   # "Capabilities" section MUST list tools
ollama run llama3.1:8b "hello"             # smoke test
```

| Spec                  | Value     |
|-----------------------|-----------|
| Disk size             | ~5 GB     |
| Inference RAM         | ~6 GB at Q4_K_M |
| Architecture          | Dense 8B (no MoE) — every parameter activates per token |
| Best for              | chat AND Goose agent tool-use (one model serves both)    |
| Tools capability      | Yes — Ollama library tag has it in the template |

This is the one model you need. The sections below list alternatives — **you can skip them on a first read** and come back if you outgrow the default.

### Air-gap transfer

Ollama doesn't ship a polished `export` command, but its on-disk format is straightforward — `manifests/` and `blobs/` subdirectories under `~/.ollama/models/`. To move the pulled model from a networked machine to an air-gapped target:

```bash
# On networked machine (after `ollama pull llama3.1:8b`)
tar czf ollama-llama3.1-8b.tgz -C ~/.ollama models/
# Tarball will be ~5 GB.

# On air-gapped target:
mkdir -p ~/.ollama
tar xzf ollama-llama3.1-8b.tgz -C ~/.ollama/
ollama serve &
ollama list             # should now show llama3.1:8b
```

If you have other Ollama models on the networked machine that you don't want to ship, look in `~/.ollama/models/manifests/registry.ollama.ai/library/llama3.1/8b` to see the SHA-256 references for the blobs you actually need, and tar only those — but for one model the whole-models-dir approach is faster than figuring that out.

### Other Ollama-library tags (reference — not pulled by default)

Any tag whose default `ollama show <tag>` output lists `tools` in the Capabilities section will work as a drop-in for Goose. Worth knowing about:

| Tag                       | Size      | Best for                                                    |
|---------------------------|-----------|-------------------------------------------------------------|
| **`llama3.1:8b`**         | ~5 GB     | **Default** — dense 8B, tools-capable, fits on any modern Mac, ~10–15 tok/s on CPU |
| `llama3.2:3b`             | ~2 GB     | Smaller + faster; tools-capable but weaker on multi-step agent loops |
| `llama3.2:1b`             | ~1.3 GB   | Smallest tools-capable Llama — useful only for smoke testing the agent path |
| `llama3.1:70b`            | ~40 GB    | Older 70B with 128K context; rock-solid tool calls, battle-tested with Ollama — only if you have ≥48 GB free RAM |
| `llama3.3:70b`            | ~43 GB    | Newest 70B, matches Llama 3.1 405B on tool use; best multi-step discipline of any local Llama. Needs ≥48 GB free RAM and ~1–3 tok/s on CPU |
| `mistral-nemo:latest`     | ~7 GB     | Mistral's tool-capable model — non-Llama alternative if you want one |

To switch, `ollama pull <tag>` and update `GOOSE_MODEL` in `configs/goose-config.yaml`. Any open `goose session` needs a restart to pick up the change.

### Going bigger — flagship-size Ollama tag

If your hardware has the RAM (≥256 GB or so), Meta's largest open release is also in the Ollama library. Same `ollama pull` + air-gap-transfer pattern applies — just a much bigger tarball.

#### Tier 2 — Llama 3.1 405B (~230 GB at Q4_K_M, server-class only)

Meta's largest openly released model — dense 405B parameters. State-of-the-art on tool use, math, and multilingual reasoning among open weights, but at this size you need a real server (≥256 GB RAM) and you'll pay a serious throughput penalty on CPU. Llama 3.3 70B was designed to match it on most benchmarks at ~5× less RAM, so reach for the 405B only if you genuinely need the last few points of quality.

| Model                          | Best for                                                         | Q4_K_M file | RAM needed |
|--------------------------------|------------------------------------------------------------------|-------------|------------|
| **`llama3.1:405b`**            | flagship open-weights model; matches frontier closed models on many tool-use evals | ~230 GB     | ~250 GB    |

Anything else from the Ollama library tagged with `tools` will also work — `mistral-nemo:latest`, `firefunction-v2`, etc. The relevant check is always `ollama show <tag> | grep -i tools` (or just `ollama show <tag>` and read the Capabilities line) before configuring Goose to use it.

### No custom Modelfile needed for `llama3.1:8b`

Earlier drafts of this project had a custom GGUF + `Modelfile` import flow for the serving model — that's gone. Library models like `llama3.1:8b` come with the right chat template (including tool-call placeholders), correct sampling defaults, and tools capability already registered. Pulling them is one command and you're done; no editing, no `ollama create`, no `FROM` path to maintain.

If you ever want to customize the library tag (bigger `num_ctx`, custom `SYSTEM` prompt, etc.), you can write a derived Modelfile:

```
# example only — not part of the default path
FROM llama3.1:8b

PARAMETER num_ctx 32768       # bigger context for agent loops
PARAMETER temperature 0.2
```

Then `ollama create my-custom-tag -f thatfile`. The derived tag inherits the library tag's template (including tools capability) and overrides only the parameters you specified.

### Context window size — how to change it, and why it's load-bearing for repo work

Ollama defaults each model to a small context window (4096 tokens on older Ollama, 8192 on recent releases) regardless of what the underlying model architecturally supports. That's fine for chat — a question plus a paragraph answer easily fits. It's *far* too small for any actual code work, because even one moderately long source file plus the agent's reasoning trace will overflow it.

#### Four ways to bump `num_ctx`, in order of permanence

1. **Per-request via the API** — one-off, doesn't persist. Useful for testing whether a larger window actually fixes the answer before committing to the higher RAM cost:
   ```bash
   curl -s http://localhost:11434/api/chat -d '{
     "model": "llama3.1:8b",
     "messages": [{"role": "user", "content": "..."}],
     "options": {"num_ctx": 32768}
   }'
   ```

2. **Inside a REPL session** — until you `/bye`:
   ```bash
   ollama run llama3.1:8b
   >>> /set parameter num_ctx 32768
   ```

3. **Baked into a derived model tag** (recommended for project use — persists across daemon restarts and is what Goose will inherit):
   ```
   # ~/Modelfile.llama-3.1-8b-32k
   FROM llama3.1:8b
   PARAMETER num_ctx 32768
   ```
   ```bash
   ollama create llama3.1-8b-32k -f ~/Modelfile.llama-3.1-8b-32k
   ```
   Then point `GOOSE_MODEL` in `~/.config/goose/config.yaml` at the new `llama3.1-8b-32k` tag (and update `configs/goose-config.yaml` in this repo if you want the change to survive a re-copy).

4. **Global daemon default** — applies to every model unless overridden by a Modelfile or request. Add to `~/.bashrc` (Linux) or `~/.zshrc` (macOS) **before** starting `ollama serve`:
   ```bash
   export OLLAMA_CONTEXT_LENGTH=32768
   ```
   Then `pkill -f ollama && ollama serve` so the daemon picks up the new env. A running daemon will not reload its environment — see Step 1's "If Ollama is already running" section for the full restart drill.

Verify what actually applied with:
```bash
ollama show <tag> --parameters    # resolved PARAMETER values, including num_ctx
```

#### Why the small default is load-bearing for real work

A useful question about a real codebase usually requires the model to *see* one or more relevant files, *plus* enough headroom to reason about them and emit an answer. Concretely:

- A single 500-line source file ≈ ~3K tokens. Two files plus a question plus an answer already overruns the 8K default.
- An agent loop (Goose-style) layers a system prompt + tool-call schemas + tool result outputs on top of the user prompt. The agent's "working memory" budget is the context window *minus* all that — often 30–50% is gone before the model even sees the first file's contents.
- "Lost in the middle" — even when content technically fits, smaller models (≤ 8B) reliably attend less well to the middle of a long context than to the start and end. Bigger models hold a long window coherently; smaller ones don't, even when they technically advertise 128K support.

For genuinely useful repo Q&A you need *both axes*:

| Axis | Why it matters |
|------|---------------|
| Big enough model | An 8B will technically accept 128K tokens but lose track of the middle; a 70B sustains long-context reasoning much better. |
| Big enough window | The relevant code, the question, the agent's reasoning, and any tool outputs all have to coexist in the window. |

#### RAM cost of `num_ctx` is not free

The KV cache grows roughly linearly with `num_ctx`, on top of the model weights. Approximate footprint (Q4_K_M weights + KV cache):

| Tag           | Native max ctx | RAM @ 8K | RAM @ 32K | RAM @ 128K |
|---------------|---------------:|---------:|----------:|-----------:|
| `llama3.2:1b` | 128K           | ~2 GB    | ~3 GB     | ~7 GB      |
| `llama3.2:3b` | 128K           | ~3 GB    | ~5 GB     | ~12 GB     |
| `llama3.1:8b` | 128K           | ~6 GB    | ~10 GB    | ~30 GB     |
| `llama3.1:70b`| 128K           | ~42 GB   | ~50 GB    | ~95 GB     |
| `llama3.3:70b`| 128K           | ~45 GB   | ~53 GB    | ~100 GB    |

Numbers are approximate (real KV cache size depends on quantization, Ollama's KV cache settings, and how aggressively `OLLAMA_KV_CACHE_TYPE` is set). The takeaway: doubling `num_ctx` roughly doubles the KV cache. On a 16 GB Mac running `llama3.1:8b`, you can comfortably go to ~32K context; pushing to 128K will start swapping.

#### When the repo doesn't fit even at 128K

128K tokens is roughly 80K–100K lines of typical source code — enough for a small library or this project's cisc187 corpus (~150K tokens, borderline), but nowhere near enough for a real production repo with thousands of files. The two existing project answers are complementary:

- **Fine-tuning (Step 4)** — bake the corpus into the weights, so the model "knows" the content without needing it in context on every query. Pays the cost once at training time, not per query.
- **Goose agent (Step 3)** — give the model filesystem tools so it can autonomously read only the files relevant to each question. Keeps the context small per turn by paging in just what's needed.

The pipeline supports both approaches: Step 4 fine-tunes the corpus into the weights; Step 3's Goose configuration gives the model live filesystem tools. Both are useful in different situations — fine-tuning when the corpus is stable and you want fast, single-prompt Q&A; Goose when the corpus changes often or you want the model to cite specific files.

---

## Operating Ollama — common commands

Reference for day-to-day use once everything is installed.

### Is the daemon running?

Four ways to check, in rough order of usefulness:

```bash
ollama ps                                   # cleanest: succeeds if daemon is up
pgrep -l ollama                             # process check
curl -s http://localhost:11434/api/tags     # HTTP check, returns JSON of installed models
systemctl status ollama                     # systemd-managed installs (Linux)
```

`ollama ps` is the friendliest — it returns the loaded-model table (possibly empty) if Ollama is up, and errors plainly if it isn't.

### Start / stop the daemon

```bash
ollama serve                                # foreground
nohup ollama serve > ~/ollama.log 2>&1 &    # detached background

systemctl start ollama                      # systemd install
systemctl stop ollama
systemctl restart ollama                    # after editing env vars in the unit
```

To stop a manually-started daemon: `Ctrl-C` if it's in the foreground, otherwise `pkill ollama`.

### List models installed on disk

```bash
ollama list
```

Output columns: tag, ID, size, modified time. This is what's stored in Ollama's blob+manifest store — **not** the same as what's currently loaded in RAM.

### See which models are loaded in RAM right now

```bash
ollama ps
```

Shows tag, ID, size, processor (CPU/GPU), and time until auto-unload. With `OLLAMA_KEEP_ALIVE=-1` set, the time column reads `Forever`.

### Inspect a specific model

```bash
ollama show llama3.1:8b                   # summary (params, arch, license)
ollama show llama3.1:8b --modelfile       # the Modelfile this tag was imported with
ollama show llama3.1:8b --parameters      # resolved PARAMETER values
ollama show llama3.1:8b --template        # chat template
```

`--modelfile` is the one you'll reach for most — it shows exactly what config Ollama is applying to that tag, including any `PARAMETER` overrides from the import.

### Add a model

Two paths:

1. **From a GGUF on disk (the air-gapped path)** — create a `Modelfile` (see Step 2), then:
   ```bash
   ollama create my-tag -f Modelfile
   ```
2. **From the Ollama registry** — `ollama pull <name>`. Requires network; will not work on the air-gapped box.

### Remove a model

```bash
ollama rm llama3.1:8b
```

Removes the named tag and decrements the refcount on the underlying GGUF blob. The blob is garbage-collected once no tag points at it. The original `.gguf` file you imported from is **not** touched.

### Manually unload a model from RAM (without removing it)

```bash
ollama stop llama3.1:8b
```

Frees RAM without deleting anything; the next request reloads it from disk. Useful when swapping between models on a tight-RAM box.

### Run a one-shot prompt

```bash
ollama run llama3.1:8b "explain what binary search is in two sentences"
```

### Open an interactive REPL

```bash
ollama run llama3.1:8b
```

Inside the REPL, useful slash commands:

- `/show parameters` — current sampling settings
- `/show modelfile` — the Modelfile this tag was imported with
- `/set parameter temperature 0.1` — change a sampling param for this session
- `/load <tag>` — switch to a different model without leaving the REPL
- `/bye` — exit

### Hit the HTTP API directly

This is what the eval and training-pipeline scripts will use:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "write hello world in Go",
  "stream": false
}' | jq -r .response
```

For multi-turn conversations use `/api/chat` with a `messages` array instead of a single `prompt` string.

---

## Step 3 — Install Goose (AI coding agent with filesystem tools)

[Goose](https://github.com/block/goose) is Block's open-source CLI coding agent — the closest offline analog to Claude Code. It gives the local Ollama model **real filesystem tools** (`list_files`, `read_file`, `run_shell`, `write_file`, plus MCP server support for extensions), so the model autonomously explores the repo as it works.

> **Why Goose and not Aider?**
> The earlier draft of this project used Aider. Aider is a fine pair-programming tool but it isn't an *agent*: it pulls file contents into the prompt itself and ships them to the LLM as text. The LLM never has tools, never autonomously reads anything, and confusingly fails when you ask it questions about content you didn't first `/add`. Goose fixes the fundamental architecture — the model gets actual tools, not a context shuttle. For the air-gapped use case (where Claude Code is unavailable), Goose's CLI shape and tool-use design are the closest offline match.

### Why Goose specifically (vs. other agent tools)

| Property | Why it fits the air-gap |
|----------|-------------------------|
| **CLI-based** (`goose session`) | Same shape as Claude Code; no IDE dependency |
| **Single static binary** (Rust) | No Python dep tree, no `pip install` chain, no PEP 668 hassles |
| **Native Ollama support** | First-class provider, no LiteLLM/openai-api middleware |
| **MCP server support** | Plug in the same MCP servers Claude Code uses |
| **Built-in tools** | filesystem, shell, code execution, file editor — exactly what's missing from Aider |
| **Air-gap friendly** | Single tarball; no runtime network needed once installed |

### Download the binary (on the networked machine)

Pick the asset matching the air-gapped target's OS and architecture:

```bash
# Linux x86_64 — most likely the air-gapped target
curl -L https://github.com/block/goose/releases/latest/download/goose-x86_64-unknown-linux-gnu.tar.bz2 \
  -o goose-linux-x86_64.tar.bz2

# Linux ARM64 (Graviton, Pi, etc.)
curl -L https://github.com/block/goose/releases/latest/download/goose-aarch64-unknown-linux-gnu.tar.bz2 \
  -o goose-linux-arm64.tar.bz2

# macOS Apple Silicon
curl -L https://github.com/block/goose/releases/latest/download/goose-aarch64-apple-darwin.tar.bz2 \
  -o goose-macos-arm64.tar.bz2

# macOS Intel
curl -L https://github.com/block/goose/releases/latest/download/goose-x86_64-apple-darwin.tar.bz2 \
  -o goose-macos-x86_64.tar.bz2
```

If a curl returns 404, the release asset name may have changed — open the latest release page at <https://github.com/block/goose/releases> and copy the actual asset URL.

### Transfer to the air-gapped target and install

```bash
# Choose a directory on PATH. ~/bin works for personal install on either OS;
# /usr/local/bin works on Linux with sudo. Use whichever matches your setup.
mkdir -p ~/bin
tar xjf goose-linux-x86_64.tar.bz2 -C ~/bin/        # adjust to the file you transferred
chmod +x ~/bin/goose

# Put ~/bin on PATH if it isn't already.
RC=~/.bashrc; [ "$(uname)" = "Darwin" ] && RC=~/.zshrc
echo 'export PATH="$HOME/bin:$PATH"' >> "$RC"
source "$RC"

goose --version
```

### Point Goose at Ollama + the Llama tag

Goose stores configuration at `~/.config/goose/config.yaml`. The repo ships a template at `configs/goose-config.yaml`:

```bash
mkdir -p ~/.config/goose
cp configs/goose-config.yaml ~/.config/goose/config.yaml
```

The template defaults to:

```yaml
GOOSE_PROVIDER: ollama
GOOSE_MODEL: llama3.1:8b            # the Ollama tag from Step 2
OLLAMA_HOST: http://127.0.0.1:11434
```

Alternatively, run `goose configure` and answer the interactive prompts — it produces an equivalent file.

### Smoke test

```bash
ollama ps                            # confirm the daemon is up
cd ~/Documents/GitHub/air_gapped_LLM
goose session                        # opens an interactive agent session in this directory
```

Inside the session, try a real test of the tool path:

```
> what's in the repo_to_fine_tune/cisc187-reader-master directory? give me a one-line summary of each subdirectory.
```

The agent should autonomously call `list_files` / `read_file` and answer based on what it actually found — no `/add` ceremony like Aider required.

### A note on throughput vs. tool-call quality

The default config uses `llama3.1:8b` — dense 8B, no reasoning traces, no expert routing. It produces tokens at ~10–15 tok/s on a mid-tier CPU box, so most agent steps complete in seconds rather than minutes. Trade-off: noticeably weaker on plans that chain four or more tool calls. If you have ≥48 GB free RAM and you find the 8B getting confused on multi-step plans, scale up to `llama3.3:70b` (or `llama3.1:70b`) — same `ollama pull` flow, also tools-capable, much slower (~1–3 tok/s) but the best tool-call discipline of any local Llama.

### Expectations

Be honest about what running Goose against a local Llama 3.1 8B on CPU gives you:

- **Throughput:** ~10–15 tokens/sec generation on a mid-tier CPU box. Most agent steps finish in 5–30 seconds.
- **Quality:** Substantially better than Aider with the same model — the agent can actually *look* at files instead of guessing. Noticeably below Claude Code, because the underlying model is much smaller than Anthropic's frontier offering.
- **Tool reliability:** Llama 3.1 8B handles single- and two-tool tasks reliably. Multi-step plans that chain 4+ tool calls drift more often than they do on the 70B. If you see format errors specifically, the usual fix is a `num_ctx` bump (the model truncated the tool schema), not a model swap.

This is the closest "Claude-Code-but-offline" experience you can realistically build today on a laptop-class machine. If you find quality unacceptable on multi-step work, the answer is "use a bigger base model" (`llama3.3:70b` if you have ≥48 GB RAM, `llama3.1:405b` only on a real server) or "loosen the air-gap to allow Claude Code access" — not "more Goose tuning."

---

## Step 4 — Install LLaMA-Factory and run a proof-of-concept fine-tune

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is a YAML-driven wrapper around `transformers` + `peft` + `accelerate` for LoRA / QLoRA / full fine-tuning. It hides the training-loop boilerplate behind a single `llamafactory-cli train <config.yaml>` command — exactly what we want for an air-gapped CPU box.

> **Training is CPU-only on this project — no GPU.** That shapes a few decisions below: we install the CPU-only PyTorch build (`https://download.pytorch.org/whl/cpu`), skip CUDA-specific deps, default `bf16` / `fp16` to `false`, and add a dedicated **CPU tuning** step (4c) before training. CPU PyTorch is *extremely* sensitive to threading and memory-allocator settings; default-out-of-the-box, a 64-core box might use 16 cores at 60% utilization. With the tuning step applied, the same hardware can train 5–10× faster. Skipping 4c is the single most common reason CPU training feels unbearably slow.

The proof-of-concept run:

- **Base model:** `Llama-3.2-1B-Instruct` — Meta's 1B instruct model. The right size for proving the pipeline end-to-end in 3–6 h instead of the ~20 h that 3B costs. Answers will be coherent but mediocre; the production scale-up to 3B is documented at the end of Step 4.
- **Dataset:** the `cisc187-reader-master` repo, prepared as continued-pretraining text.
- **Method:** LoRA continued pretraining (`stage: pt` in LLaMA-Factory terms).
- **Success criterion:** the pipeline runs to completion — dataset prep → train → merge → convert to GGUF → import to Ollama → query. A 3B base trained on the cisc187 corpus should produce genuinely usable Q&A (much better than the smaller models we considered earlier), so the success bar moves up from "any coherent response" to "answers that reference the textbook's specific framing on most questions."

> **Two important constraints up front:**
>
> 1. **Fine-tuning needs the HuggingFace-format model**, not the GGUF you downloaded in Step 2. They are two *different* files of the same model:
>
>    | File from | Format | Size | Used by |
>    |-----------|--------|------|---------|
>    | Step 2 (`curl`) | GGUF Q4_K_M (quantized, 4-bit) | ~400 MB | Ollama inference only |
>    | Step 4 (`huggingface-cli`) | HF safetensors FP16 (full precision) | ~1 GB | LLaMA-Factory training |
>
>    You can't fine-tune from GGUF: it's a one-way export optimized for inference (memory-mapped, quantized, no PyTorch module structure for backprop), and Q4 weights aren't trainable on CPU anyway — you need FP16. After training, the pipeline produces a *new* GGUF from the merged weights, which is what Ollama then serves.
> 2. **Stop Ollama before training.** A loaded model in Ollama's RAM will fight LLaMA-Factory for memory and CPU bandwidth. `pkill ollama` (or `systemctl stop ollama`) before launching a training run, restart after.

### 4a. On the networked machine — gather everything

#### Pick a base model to fine-tune

The training pipeline accepts any HuggingFace-format causal LM. The **current default is Llama-3.2-1B-Instruct** — the entire walkthrough below (YAML paths, Modelfile, GGUF filename, Ollama tag) is set up for it. The 3B is the natural production scale-up after the POC succeeds; the 8B and 70B are server-class options.

| Size class                          | Recommended HF path                       | Best for                                                                |
|-------------------------------------|-------------------------------------------|-------------------------------------------------------------------------|
| **Default (Mac POC)**               | `unsloth/Llama-3.2-1B-Instruct`           | **The walkthrough below uses this.** ~1–2 h/epoch on Mac CPU, 5 epochs ≈ ~3–6 h. Validates the pipeline end-to-end; coherent-but-mediocre Q&A quality. |
| Production (Mac CPU)                | `unsloth/Llama-3.2-3B-Instruct`           | Scale-up after POC. ~4–5 h/epoch, 3 epochs ≈ ~20 h on i9-class Mac. Real Q&A quality. |
| Bigger — Linux target               | `unsloth/Meta-Llama-3.1-8B-Instruct`      | ~18 h/epoch on Mac CPU (impractical for iteration). On a 32+ core Linux box: ~6–8 h/epoch. Production-grade quality. |
| Server-class only                   | `unsloth/Llama-3.3-70B-Instruct`          | Needs ≥96 GB RAM (FP16 weights + LoRA optimizer state + activations) and multi-day patience. Only attempt on a real server. |

> **Why `unsloth/...` paths instead of `meta-llama/...`?** Meta's official `meta-llama/...` repos are gated. `hf download meta-llama/Llama-3.2-1B-Instruct` returns `401 GatedRepoError` until you (a) request access on each model's HF page, (b) accept the Llama Community License, and (c) `hf auth login` with a read token. The `unsloth/...` namespace re-uploads the same safetensors verbatim and is ungated, so a fresh machine can `hf download` immediately. For an air-gapped POC the unsloth mirror is the path of least resistance; for a strict reproducibility audit, use the `meta-llama/...` paths and pay the auth tax (request access → wait for approval → `hf auth login` → re-run the download).

**Why Llama-3.2-1B as the POC default** — A POC validates *plumbing*, not *quality*: does the data prep produce correct JSONL, does LLaMA-Factory accept the config, does training run to completion with a steadily-dropping loss curve, does merge → GGUF → Ollama import all work, does the resulting tag answer a prompt with coherent text. All of that is checked just as well by a 1B as by a 3B, but the 1B run completes in 3–6 h instead of ~20 h on a Mac CPU. That order-of-magnitude faster feedback loop is the whole reason to start at 1B.

**Why Llama-3.2-3B for production** — Meta's instruction tuning on the 3.2 line is heavy, which translates to stable Q&A behavior after CPT (the LoRA learns the textbook content while the underlying instruct tuning keeps the model formatting answers like an assistant). The cisc187 corpus is *prose about code*, not raw code, so a strong general instruct base handles it better than a code-specialized one. The 3B is also the largest dense Llama that fine-tunes in under a day on a Mac CPU — anything bigger (8B, 70B) requires Linux. Once the 1B POC has proven the pipeline, the natural next step is to re-run the same recipe with the 3B base; the swap instructions are in the `configs/cisc187_pt.yaml` header comment.

**On smaller-than-1B Llama** — Meta doesn't publish anything below 1B in the Llama 3.x line. `unsloth/Llama-3.2-1B-Instruct` is the smallest official-equivalent option; non-Llama small models exist but are outside this Llama-only pipeline.

To use a model other than the current POC default, change the `hf download <hf-path>` line in the bash block below to your chosen path, and update `model_name_or_path` in `configs/cisc187_pt.yaml` to match the local-dir name.

#### What gets bundled

You'll bundle four things for transfer:

- LLaMA-Factory source (gives you example configs + the `export` command)
- Python wheel cache for LLaMA-Factory and its deps (CPU-only torch)
- The base model (POC default `Llama-3.2-1B-Instruct`; swap to `Llama-3.2-3B-Instruct` for the production run after the POC succeeds) in HuggingFace format
- llama.cpp source (for the `convert_hf_to_gguf.py` script we'll run after training)

```bash
mkdir llf-bundle && cd llf-bundle

# 1. LLaMA-Factory source (shallow clone — we don't need history)
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# 2. Python wheels (CPU-only torch from the dedicated PyTorch index).
#
#    Note: we install with just [torch], NOT [torch,metrics]. The [metrics]
#    extra pulls jieba (Chinese word segmenter, for BLEU/ROUGE scoring), and
#    jieba is only published as an sdist on PyPI — incompatible with our
#    --only-binary=:all: requirement. We don't use BLEU/ROUGE in this
#    pipeline — quality is judged by manual inspection of model output via
#    `ollama run`. If you ever want [metrics]
#    back, add `--no-binary=jieba` alongside `--only-binary=:all:` — jieba
#    is pure Python so the sdist installs without a compiler.
mkdir llf-wheels
python3.11 -m pip download \
  --dest ./llf-wheels \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "llamafactory[torch]" "setuptools<81"
# One pin worth understanding:
#
# - setuptools<81: LLaMA-Factory's mm_plugin.py indirectly triggers librosa,
#   which still imports `from pkg_resources import ...`. Setuptools 81+
#   removed pkg_resources entirely, so we pin <81 to keep it available.
#   setuptools 80 still works fine for everything else.
#
# Note on torch: we deliberately do NOT pin a minimum torch version. pip
# picks the latest torch wheel that exists for the host platform — on Linux
# x86_64 that's recent torch with all APIs LLaMA-Factory uses; on older
# platforms (e.g. macOS Intel, where torch caps at 2.2.2) pip falls back
# to whatever still ships wheels. Pinning torch>=2.4 was tried and removed
# because it broke installs on platforms that legitimately can't reach it.

# 3. Llama-3.2-1B-Instruct in HuggingFace format (~2.5 GB; ~6 GB for 3B,
#    ~16 GB for 8B, ~140 GB for 70B).
#    This is NOT a duplicate of the Step 2 download — that one was the GGUF
#    (quantized, Ollama-only); this is the FP16 safetensors + tokenizer
#    files that LLaMA-Factory needs to actually fine-tune.
#
#    The unsloth/ mirror is ungated — no `hf auth login` required. If you
#    prefer Meta's official meta-llama/ repos, request access on the HF
#    model page first, then `hf auth login`, then swap unsloth/ → meta-llama/
#    in the command below.
python3.11 -m pip install --user huggingface_hub                # only on networked machine
hf download unsloth/Llama-3.2-1B-Instruct \
  --local-dir Llama-3.2-1B-Instruct

# 4. llama.cpp source (we only need convert_hf_to_gguf.py — shallow clone is fine)
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
```

If the networked machine differs in OS/arch from the air-gapped target, use the same Docker-container or `--platform` trick documented in Step 3 — wheel-platform mismatches will bite the same way.

Bundle everything:

```bash
cd ..
tar czf llf-bundle.tgz llf-bundle/
```

You should also have `cisc187-reader-master/` available — either bundle it here too, or transfer separately.

### 4b. Transfer to the air-gapped box, install LLaMA-Factory

```bash
# Extract into your HOME directory. This matters because Steps 4d, 4e, and 4g
# all use absolute paths under ~/llf-bundle/ (datasets, LLaMA-Factory, the HF
# model dir, llama.cpp source). If you extract somewhere else, every later
# step's paths break. Adjust the tgz source path to wherever you copied it.
cd ~
tar xzf ~/Downloads/llf-bundle.tgz       # produces ~/llf-bundle/
cd ~/llf-bundle

# 1. Install LLaMA-Factory and its deps into user-site.
python3.11 -m pip install --user --no-index --find-links ./llf-wheels "llamafactory[torch]"

# 2. Pin setuptools to <81 in user-site so pkg_resources stays available.
# Setuptools 81 removed pkg_resources, and llamafactory's mm_plugin.py
# eagerly imports librosa, which still uses pkg_resources. Without this
# step, `llamafactory-cli version` blows up with "ModuleNotFoundError:
# No module named 'pkg_resources'". --force-reinstall is needed because
# if the system already has setuptools 82, plain --user is a no-op.
python3.11 -m pip install --user --force-reinstall --no-index --find-links ./llf-wheels "setuptools<81"

# 3. Put the user-site CLI scripts on PATH so `llamafactory-cli` resolves.
# Safe to re-run; idempotent. Use ~/.zshrc on macOS, ~/.bashrc on Linux.
export PATH="$(python3.11 -m site --user-base)/bin:$PATH"
echo 'export PATH="$(python3.11 -m site --user-base)/bin:$PATH"' >> ~/.zshrc      # macOS
# echo 'export PATH="$(python3.11 -m site --user-base)/bin:$PATH"' >> ~/.bashrc   # Linux

llamafactory-cli version
```

If you also want LLaMA-Factory's source-installed editable mode (lets you tweak training-loop internals):

```bash
cd LLaMA-Factory
python3.11 -m pip install --user --no-index --find-links ../llf-wheels -e ".[torch]"
```

#### If step 2 of 4b doesn't stick — `ModuleNotFoundError: No module named 'pkg_resources'`

If you ran step 2 above (`pip install --user --force-reinstall "setuptools<81"`) and `llamafactory-cli version` still hits this error, the system's setuptools 82+ is still winning the `sys.path` lookup. Verify which one is being imported:

```bash
python3.11 -c "import pkg_resources; print(pkg_resources.__file__)"
# Want:  ~/Library/Python/3.9/.../site-packages/pkg_resources/__init__.py   (macOS)
# or:    ~/.local/lib/python3.11/site-packages/pkg_resources/__init__.py    (Linux)
```

If the path is in `/usr/local/...` (or returns `ModuleNotFoundError`), the system version is shadowing user-site. Uninstall it, then reinstall <81 into user-site:

```bash
python3.11 -m pip uninstall -y setuptools
python3.11 -m pip install --user --no-index --find-links ./llf-wheels "setuptools<81"
llamafactory-cli version
```

**Background:** `pkg_resources` used to ship as a sibling top-level module alongside `setuptools`. **Setuptools 81 dropped it entirely.** LLaMA-Factory's `mm_plugin.py` eagerly imports `librosa`, which still uses `from pkg_resources import resource_filename`. Any system with setuptools 81+ (or no setuptools at all) will fail until an older setuptools is back in the import path. That's why step 2 of 4b is a regular part of the install, not just a contingency.

### 4c. CPU tuning — actually use those cores

PyTorch CPU training is bottlenecked by threading and the memory allocator, not the framework. Default settings often leave most cores idle. Five knobs, roughly in order of impact:

#### 1. Threading env vars (biggest single win)

Find your **physical** core count — hyperthreaded siblings contend for the same vector units and slow training down. Use `lscpu`:

```bash
lscpu | grep -E "^Socket|^Core|^Thread"
# Socket(s):           1
# Core(s) per socket:  32
# Thread(s) per core:  2     ← ignore the logical count
```

Physical cores = `Sockets × Cores per socket`. Add to `~/.bashrc`, replacing `32` with yours:

```bash
# Match physical core count from lscpu — do NOT use logical (hyperthreaded) count.
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

# Pin OpenMP/MKL threads to physical cores. Big NUMA / HT win on Intel.
export KMP_AFFINITY=granularity=fine,compact,1,0

# Keep threads spinning briefly between parallel regions instead of resleeping.
export KMP_BLOCKTIME=1
```

`source ~/.bashrc` after editing.

#### 2. Memory allocator (10–30% speedup, almost free)

PyTorch's default allocator fragments badly over a long training run. Preload tcmalloc or jemalloc — easiest standalone optimization on CPU. Install on the networked machine and stash the `.deb`/`.rpm` in the bundle (these are system packages, not Python):

```bash
# tcmalloc (from google-perftools)
sudo apt-get install google-perftools                                  # Debian/Ubuntu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# OR jemalloc (often slightly better on AMD)
sudo apt-get install libjemalloc2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

Add the chosen `LD_PRELOAD` line to `~/.bashrc` so it applies to every training run.

#### 3. Intel Extension for PyTorch — IPEX (Linux x86_64 + Intel CPU only)

[IPEX](https://github.com/intel/intel-extension-for-pytorch) unlocks AMX BF16 kernels on Sapphire Rapids / Emerald Rapids and bundles a launcher (`ipexrun`) that auto-tunes threading, allocator, and core pinning in one shot — replacing knobs 1 + 2 if you use it.

> **IPEX is Linux-x86_64-only.** Intel doesn't publish wheels for macOS, Windows, or ARM. If you `python3.11 -m pip download intel_extension_for_pytorch` from a Mac, pip silently drops it (no wheel matches the host platform) and your bundle won't include it. Two ways to handle this:
>
> - **Skip IPEX entirely.** You're on Mac/ARM/AMD, or your Intel CPU is old enough that AMX isn't there anyway. Knobs 1 + 2 + 4 still give meaningful speedups; you just lose AMX.
> - **Build Linux wheels from the Mac.** Add `--platform manylinux2014_x86_64 --abi cp311 --python-version 311 --implementation cp` to the `python3.11 -m pip download`, or run the download inside `python:3.11-slim` via Docker. Either way produces a bundle that works on the Linux training box.

On the networked machine, if appropriate for the target, add IPEX to the wheel set:

```bash
python3.11 -m pip download \
  --dest ./llf-wheels \
  --only-binary=:all: \
  intel_extension_for_pytorch
```

On the air-gapped box (same Python environment as LLaMA-Factory):

```bash
python3.11 -m pip install --user --no-index --find-links ./llf-wheels intel_extension_for_pytorch
```

If you see `ERROR: Could not find a version that satisfies the requirement intel_extension_for_pytorch`, the wheel cache doesn't contain a wheel for your platform — that's expected on Mac/ARM/AMD. Skip this step; the rest of the pipeline runs without it.

Check whether your CPU actually has AMX:

```bash
grep -o 'amx[_a-z0-9]*' /proc/cpuinfo | sort -u
# Look for: amx_tile, amx_int8, amx_bf16
```

If AMX is present, you can safely set `bf16: true` in the training YAML — substantial speedup. **Without AMX, leave `bf16: false`** (emulated BF16 is slower than fp32).

#### 4. NUMA pinning (multi-socket boxes only)

If `lscpu` shows `Socket(s): 2` or more, training threads crossing sockets thrash memory bandwidth. Pin training to one NUMA node:

```bash
sudo apt-get install numactl                              # one-time, on the target

# Wrap the training command at launch time (see 4f):
numactl --cpunodebind=0 --membind=0 llamafactory-cli train configs/cisc187_pt.yaml
```

Counterintuitively, restricting training to one socket is usually *faster* than spreading across both, because cross-socket memory access is the killer.

#### 5. AMD CPUs

ZenDNN exists as an AMD equivalent of IPEX, but its PyTorch integration is finicky. The threading-env-var step (1), the allocator step (2), and NUMA pinning (4) carry the bulk of the gain on AMD — skip step 3 entirely.

#### Verify the tuning is actually applied

During the first training step, `htop` (or `top` with `1` pressed to show per-core) should show **all** `$OMP_NUM_THREADS` cores pegged near 100%. If only a handful are loaded, the env vars haven't reached the training process — usually because `.bashrc` wasn't reloaded, or you launched in a stale shell.

### 4d. Prepare the dataset (cisc187-reader-master → CPT JSONL)

LLaMA-Factory's continued-pretraining stage expects one JSON record per training sample, with a `text` field. The repo's `scripts/prepare_cisc187.py` walks the source repo, includes `.rst` / `.txt` / `.py` / `.cpp` / `.h` / etc., drops generated docs and build state, and writes one JSONL record per file (prefixed with the relative path as a weak "where did this come from" signal).

Place the cisc187 corpus at `~/llf-bundle/datasets/cisc187-reader-master/`, then run:

```bash
python3.11 scripts/prepare_cisc187.py \
  --repo ~/llf-bundle/datasets/cisc187-reader-master \
  --out  ~/llf-bundle/datasets/cisc187_pt.jsonl
```

On the included copy of the reader, this writes ~205 records.

Register the dataset with LLaMA-Factory by adding the entry from `configs/dataset_info_patch.json` to `LLaMA-Factory/data/dataset_info.json` (keep all existing entries; just add this one). The patch file contains:

```json
"cisc187_pt": {
  "file_name": "/REPLACE/WITH/ABSOLUTE/PATH/TO/datasets/cisc187_pt.jsonl",
  "columns": {
    "prompt": "text"
  }
}
```

Replace the placeholder with the absolute path to the JSONL on the air-gapped box. The absolute path matters — LLaMA-Factory resolves dataset paths from its own root unless given an absolute one.

### 4e. Training config

The repo ships `configs/cisc187_pt.yaml` ready to use, with **3 placeholder paths you have to replace before training**. Open the file and edit these three lines (everything else can stay as-is):

| Line  | Field                | Currently                                                              | Replace with                                                                                                |
|-------|----------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| ~9    | `model_name_or_path` | `/REPLACE/WITH/PATH/TO/Llama-3.2-1B-Instruct`                    | `/Users/<you>/llf-bundle/Llama-3.2-1B-Instruct` (macOS) or `/home/<you>/llf-bundle/...` (Linux)        |
| ~27   | `dataset_dir`        | `/REPLACE/WITH/PATH/TO/llf-bundle/LLaMA-Factory/data`                  | `/Users/<you>/llf-bundle/LLaMA-Factory/data` (macOS) or `/home/<you>/llf-bundle/LLaMA-Factory/data` (Linux)  |
| ~34   | `output_dir`         | `/REPLACE/WITH/PATH/TO/llf-bundle/output/llama-3.2-1b-cisc187-lora`       | `/Users/<you>/llf-bundle/output/llama-3.2-1b-cisc187-lora` (macOS) or `/home/<you>/llf-bundle/output/...` (Linux) |

All three must be **absolute** (start with `/`). LLaMA-Factory does not expand `~` or resolve relative paths for these fields — a literal `~/...` or `./...` is passed straight to `open()` and fails with `FileNotFoundError`. Use `/Users/...` on macOS, `/home/...` on Linux.

The `output_dir` in particular matters for Step 4g: that's where the trained LoRA adapter is saved. If `output_dir` is relative (`./output/...`), the adapter lands wherever you happened to be when you ran training — and Step 4g's `llamafactory-cli export` will fail with `ValueError: Can't find 'adapter_config.json'` because it looks at the path you give it, which won't match the relative-CWD location. Making `output_dir` absolute pins the adapter to a known location under `~/llf-bundle/` so the export step always finds it.

You also need to edit **1 line** in `configs/dataset_info_patch.json` (the `file_name` value), then paste that block into `~/llf-bundle/LLaMA-Factory/data/dataset_info.json` per Step 4d.

For reference, the ready-to-edit `configs/cisc187_pt.yaml` looks like this — the two `/REPLACE/WITH/...` lines are what you change:

```yaml
### model
model_name_or_path: /REPLACE/WITH/PATH/TO/Llama-3.2-1B-Instruct
trust_remote_code: false

### method
stage: pt                        # continued pretraining (next-token loss on raw text)
do_train: true
finetuning_type: lora
lora_target: all                 # apply LoRA to all linear layers; safer than guessing
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: cisc187_pt              # must match the key in dataset_info.json
dataset_dir: /REPLACE/WITH/PATH/TO/llf-bundle/LLaMA-Factory/data
cutoff_len: 1024                 # keep sequences short on CPU
max_samples: 100000              # cap if your repo is huge; otherwise harmless
preprocessing_num_workers: 4

### output
output_dir: ./output/llama-3.2-1b-cisc187-lora
overwrite_output_dir: true
logging_steps: 10
save_steps: 200
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8   # effective batch size 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: false                      # flip to true only on Sapphire Rapids / EPYC Genoa+
fp16: false                      # leave both off → trains in fp32 (safest CPU default)
```

Tune `bf16: true` later if your CPU supports it — significant speedup, but on older Xeons it falls back to fp32 anyway and the flag is just noise.

### 4f. Train

Stop Ollama first to free RAM and CPU:

```bash
pkill ollama        # or systemctl stop ollama
```

Then launch training. **Training goes through the `scripts/run_train.py` wrapper, not `llamafactory-cli` directly.** The wrapper adds a tiny shim for `torch.mps.device_count`, which LLaMA-Factory 0.9.x calls unconditionally during device detection. The method only exists in newer torch builds; on older ones (e.g. torch 2.2.2, which is the cap on macOS Intel since PyTorch dropped x86_64 macOS support) the call raises `AttributeError` and training crashes before reading the YAML. The shim is a no-op when the method is already present, so the wrapper is safe to use everywhere — Linux, Apple Silicon, Intel Mac — and is the documented entry point going forward.

Launch from the `air_gapped_llm` repo root. The YAML in `configs/cisc187_pt.yaml` sets `dataset_dir: ~/llf-bundle/LLaMA-Factory/data` explicitly, so LLaMA-Factory finds the edited `dataset_info.json` no matter where you `cd` from — no need to be inside the LLaMA-Factory directory to launch.

Pick the variant matching your CPU tuning from 4c:

```bash
cd ~/air_gapped_llm

# Baseline — relies on env vars from 4c (steps 1 + 2)
python3.11 scripts/run_train.py configs/cisc187_pt.yaml

# Intel + IPEX (4c step 3) — auto-tunes threading, allocator, AMX
ipexrun python3.11 scripts/run_train.py configs/cisc187_pt.yaml

# Multi-socket box (4c step 4) — pin to one NUMA node
numactl --cpunodebind=0 --membind=0 python3.11 scripts/run_train.py configs/cisc187_pt.yaml

# Both: IPEX + NUMA
numactl --cpunodebind=0 --membind=0 ipexrun python3.11 scripts/run_train.py configs/cisc187_pt.yaml
```

Expect training to take **roughly 4–6 hours per epoch on a 16–64 core box** for a 3B model + cisc187-sized corpus, assuming 4c is applied. With the default 3 epochs in the YAML, plan for ~12–18 hours total on Linux, ~20–24 hours on a Mac. Without 4c, expect 5–10× longer. Monitor:

```bash
tail -f output/llama-3.2-1b-cisc187-lora/trainer_log.jsonl
```

Loss should drop steadily for the first chunk then plateau. If it doesn't drop at all, the dataset prep is wrong (most commonly: empty JSONL or wrong `columns` mapping).

### 4g. Merge LoRA, convert to GGUF, import to Ollama

Three commands plus a Modelfile.

**Merge the adapter into the base weights:**

```bash
# All three location flags must be ABSOLUTE paths — `~/...` and relative
# paths are passed straight to open() without expansion (same constraint
# as the YAML in Step 4e). The --adapter_name_or_path here MUST match the
# absolute `output_dir` you set in the YAML.
llamafactory-cli export \
  --model_name_or_path /home/<you>/llf-bundle/Llama-3.2-1B-Instruct \
  --adapter_name_or_path /home/<you>/llf-bundle/output/llama-3.2-1b-cisc187-lora \
  --export_dir /home/<you>/llf-bundle/output/llama-3.2-1b-cisc187-merged \
  --export_size 2 \
  --export_legacy_format false
```

Output is a full HuggingFace-format model in `output/llama-3.2-1b-cisc187-merged/`.

**Convert to GGUF** (using llama.cpp's converter — runs in the same Python environment since dependencies overlap with LLaMA-Factory's):

```bash
cd ~/llf-bundle/llama.cpp
python3.11 -m pip install --user --no-index --find-links ../llf-wheels -r requirements.txt

python3.11 convert_hf_to_gguf.py \
  ../LLaMA-Factory/output/llama-3.2-1b-cisc187-merged \
  --outfile ~/llama-3.2-1b-cisc187-f16.gguf \
  --outtype f16
```

For 1B at FP16, the merged GGUF is ~2.5 GB — small enough that the FP16 file is fine to use directly via Ollama (the inference RAM cost is dominated by KV cache, not weights, at this size). Quantizing to Q4_K_M would bring it down to ~700 MB but is unnecessary for the POC. For 3B / 8B / 70B the FP16 file is much larger (~6 / ~16 / ~140 GB) and quantization becomes worthwhile — build llama.cpp's `llama-quantize` binary and run `llama-quantize <input.gguf> <output.gguf> Q4_K_M` after the convert step.

**Import to Ollama:**

Restart Ollama (`ollama serve` or `systemctl start ollama`), then use `modelfiles/Modelfile.llama-3.2-1b-cisc187`. If you registered an earlier baseline pointing at the *base* Llama-3.2-1B GGUF for A/B comparison, this same Modelfile gets re-used in Phase 2: edit just the `FROM` line so it now points at the merged GGUF:

```
FROM ~/llama-3.2-1b-cisc187-f16.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0.4
PARAMETER top_p 0.9

SYSTEM """..."""   # leave the SYSTEM line untouched — see below
```

Keep the `SYSTEM` line identical to the baseline registration. Reusing the same SYSTEM prompt across both phases means any difference you observe between the baseline and tuned answers is attributable to weight changes from fine-tuning, not to a prompt change. The Modelfile in `modelfiles/Modelfile.llama-3.2-1b-cisc187` is designed for exactly this two-phase swap — its header comment walks through both states.

Re-import using the same tag:

```bash
ollama create llama-3.2-1b-cisc187 -f modelfiles/Modelfile.llama-3.2-1b-cisc187   # replaces the baseline manifest
ollama run llama-3.2-1b-cisc187 "explain what main() does in the cisc187 reader"
```

Ollama replaces the existing `llama-3.2-1b-cisc187` manifest in-place, so any previous baseline pointing at this tag is overwritten by the tuned model.

### 4h. What "success" looks like for this POC

The proof-of-concept passes if:

- `llamafactory-cli train` completes a full epoch without errors.
- The loss curve actually decreases (sanity check that learning is happening — starts ~2.5, should drop steadily toward ~1.0–1.5 by epoch 5).
- The merged model exports and converts to GGUF cleanly.
- `ollama run llama-3.2-1b-cisc187 "<any prompt>"` returns a coherent response — even a wrong one is fine for the POC; we just need the pipeline plumbing proven.

At 1B, the fine-tune may or may not measurably outperform the untuned base on cisc187-specific framing — 1B has limited memorization capacity, so some textbook content will get baked in but a lot won't. **That's expected and fine for a POC.** The real Q&A quality test happens after the 3B production scale-up (see the swap instructions in `configs/cisc187_pt.yaml`'s header). If after the 1B training the responses look identical to the untuned baseline, re-check the loss curve in `trainer_log.jsonl` (it should drop noticeably from initial values; if it stayed flat, the data pipeline or learning rate is the problem, not the model size).

---

## Hardware requirements — full breakdown

This pipeline is intentionally **CPU-first** because the original air-gap target had no GPUs. CPU works for everything up to ~8B parameters on a serious workstation; 70B is realistic on CPU only on a server-class box with ≥128 GB RAM and you're willing to wait days. The hardware needed scales roughly linearly with the model size you want to train.

### Why RAM is the dominant constraint

LoRA fine-tuning's peak memory is roughly:

```
peak_RAM ≈ (model_weights_at_FP16) + (LoRA_optimizer_state) + (activations_at_cutoff_len) + (OS_overhead)
```

For Llama 3.2 1B at FP32 (CPU default in this pipeline) that's:

- Model weights: 1.0B × 4 bytes = ~4 GB
- LoRA optimizer state (Adam, rank 32): ~50 MB (tiny — adapter is small)
- Activations at `cutoff_len: 2048`: ~1–2 GB
- OS overhead + Python: ~1 GB
- **Total: ~6–8 GB minimum, 12 GB comfortable**

For 3B FP32: ~12 GB weights + ~3 GB activations → ~24 GB recommended free RAM. For 8B FP32: ~32 GB weights → ~64 GB free RAM. For 70B FP32: ~280 GB weights → impossible on consumer hardware without quantization-aware training.

If you can flip `bf16: true` (Intel Sapphire Rapids+, EPYC Genoa+, or any Ampere+ GPU), every memory number above halves. CPU bf16 also speeds training up by ~30–40% on AMX-capable CPUs.

### Tier-by-tier requirements

| Tier                | Free RAM   | Physical cores | CPU example                              | Fine-tune feasible up to                  | Wall-clock for 3 epochs |
|---------------------|-----------:|---------------:|------------------------------------------|-------------------------------------------|--------------------------|
| **POC laptop**      | ~8 GB     | 4–8            | 2019 i9 MacBook, M2/M3 Mac, mid-range Ryzen | Llama-3.2-1B                            | 5 epochs ≈ 6–12 h        |
| **Comfortable Mac** | ~24 GB     | 8–12           | M3 Pro / M3 Max 64 GB, 64 GB Intel desktop | Llama-3.2-3B                            | ~20 h                    |
| **Workstation**     | ~96 GB     | 32             | Threadripper 7980X (64-core), Xeon W9    | Llama-3.1-8B                            | 1–2 days                 |
| **Heavy workstation** | ~192 GB | 64             | Threadripper 9980X, Xeon Sapphire Rapids | Llama-3.1-8B comfortably; 70B with bf16 + tight packing | Multi-day for 70B |
| **Multi-GPU server**| see GPU table below | n/a   | A100/H100 cluster                         | Llama-3.3-70B in hours, not days        | Hours                    |

> **The 64 GB / 16-thread 2019 i9 MacBook in the project's POC trained Llama-3.2-1B in ~10 h.** That's the realistic baseline. Newer Apple Silicon (M3 Max / M4 Max) gets you 30–50% faster on the same model.

### Inference is cheaper than training

A model that *trains* on 12 GB RAM will *answer prompts* via Ollama in ~1–2 GB once quantized to Q4_K_M. Inference RAM ≈ (quantized model size) + (KV cache at chosen `num_ctx`). The 1B fine-tune produces a ~2.5 GB FP16 GGUF which Ollama loads in ~3 GB total. Don't conflate the two — a Mac that's tight on training RAM can comfortably *use* the trained model afterward.

---

## Recommended prebuilt machines

The right choice depends on which model size you actually want to train. All prices in USD as of 2026; rentals are per-hour cloud prices. **None of these are required — the pipeline runs on any modern Mac.** This is just guidance if you're buying hardware specifically for this work.

### Tier 1 — POC and small-model fine-tuning (≤ 3B)

**Any 32–64 GB Mac you already own.** A 64 GB M3 Pro / M3 Max / Intel i9 MacBook handles 1B and 3B fine-tunes overnight. Zero additional spend if you already have one. This is the right choice if your goal is to prove the pipeline.

If buying new for this tier specifically:
- **Mac mini M4 Pro 64 GB** — ~$2,200. Quietest, lowest power, capable up to 3B. The "I want to do this on the same desk as my regular work" choice.

### Tier 2 — Production Mac (3B–8B inference + 3B training)

**Mac Studio M3 Ultra with 192 GB RAM** — starts at $3,999, fully loaded ~$9,500. 32-core CPU, 80-core GPU, 800 GB/s unified memory bandwidth. The 192 GB unified-memory configuration is purpose-built for local LLM work: you can QLoRA-tune a 70B model locally on M3 Max 128 GB ([source](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/)), and the M3 Ultra runs 120B+ models or loads multiple 70B models simultaneously ([source](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/)). Throughput is ~1–3 tok/s for training (~30–50 on H100), so use it for inference and small-model training, rent cloud GPUs for serious 70B training runs.

### Tier 3 — Workstation for 8B+ CPU training

**BIZON X4000 / Lambda Workstation / similar Threadripper builds** — $5,000–$8,000 starting, configurable. AMD Threadripper 7980X (64 cores / 128 threads, $4,500 CPU alone) is the workhorse for CPU-side LLM work ([BIZON](https://bizon-tech.com/bizon-x4000.html), [VRLA Tech](https://vrlatech.com/vrla-tech-workstations/amd-ryzen-threadripper-9980x-processor/)). 128 PCIe 5.0 lanes mean you can add GPUs later without bottlenecking. 128 GB DDR5 ECC RAM is the recommended starting point for AI workloads, with 256 GB+ for workloads that load full datasets ([servethehome](https://www.servethehome.com/amd-ryzen-threadripper-7980x-review-a-funky-workstation-cpu-some-will-love/)).

- **BIZON X4000 G4** — starts at $4,990, configurable up to 256 GB RAM and multiple GPUs. Quiet office-tier cooling.
- **DIY equivalent** — ~$6,000 self-built (Threadripper 7980X + 256 GB ECC DDR5 + WRX90 motherboard + decent NVMe). Cheaper but you assemble.

### Tier 4 — Single GPU for 7B–13B training and 1B–8B inference

**A consumer GPU drops fine-tune time from days to hours.** The sweet spot for this project:

- **RTX 4090 (24 GB VRAM)** — ~$1,800 used, $2,500 new. QLoRA on 7B/13B in 24 GB is a solved problem with documented workflows where runs complete in hours ([craftrigs](https://craftrigs.com/guides/fine-tuning-7b-llm-consumer-gpu-unsloth-lora/)). 14B fits with tight settings.
- **RTX 5090 (32 GB VRAM)** — ~$2,500–$3,000. The extra 8 GB makes a real difference: Llama 3.1 13B at FP16 with LoRA adapters fits without hitting VRAM limits, whereas the 4090 needs INT4 ([Spheron](https://www.spheron.network/blog/rtx-5090-vs-rtx-4090/)). QLoRA on 14B sits comfortably with ~12 GB headroom.
- **Used RTX 3090 (24 GB)** — ~$700–$900 used. Identical VRAM to the 4090, ~60% the throughput. The budget pick if you have one machine to dedicate to this.

Pair any of these with a 12-core+ CPU + 64 GB RAM and you have a complete fine-tune rig for $3,000–$5,000 total.

### Tier 5 — Multi-GPU server for 70B training

**Out of laptop / workstation territory — this is renting cloud or buying server hardware.**

- **Cloud rental (8× H100 cluster)** — $16.80–$19.20 per GPU-hour on-demand ([Jarvislabs](https://jarvislabs.ai/blog/h100-price)). A 70B fine-tune at FP16 needs the full 8-card cluster to distribute the 1.1 TB memory footprint ([Lyceum](https://lyceum.technology/magazine/which-gpu-for-fine-tuning-70b-model/)). Typical 70B fine-tune total cost: $10,000–$50,000 using 8 H100s for 300–1,000 hours ([introl](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)).
- **Cloud rental (single A100 80GB)** — $1.49/hr ([Jarvislabs](https://jarvislabs.ai/blog/a100-price)). With QLoRA you can fit a 70B fine-tune on one A100 — slower but ~10× cheaper than 8× H100.
- **Buying** — A100 80GB: $7,000–$15,000 new, $4,000–$9,000 used ([introl](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)). H100: $30,000+ new. Realistically only worth buying if you'll use them constantly; otherwise rent.

For a one-time 70B fine-tune on this project's textbook-sized corpus, **rent one A100 80GB for ~$50–$100 total**. Don't buy a server.

### Decision tree

```
Is your hardware already 32 GB+ RAM Mac/Linux?
├── Yes → start with the POC (Llama-3.2-1B), no purchase needed
│   └── POC passes? Stay on this hardware for 3B production.
│       └── Need 8B+? Buy Tier 3 workstation OR rent Tier 4 GPU OR rent cloud.
│
└── No (8–16 GB RAM) → buy Tier 1 (used Mac mini M4 Pro 64 GB ≈ $2,200)
    OR start small with `llama3.2:1b` Ollama serving and skip fine-tuning entirely.
```

---

## Running on GPU — full walkthrough

The default config is CPU-only. To run on GPU, three things change.

### 1. Reinstall PyTorch with CUDA wheels

The CPU-only torch in `llf-wheels/` won't see a GPU. Replace it:

```bash
# Pick the cu* version matching your CUDA driver — check with `nvidia-smi`.
# cu124: CUDA 12.4+ (H100, 5090, 4090 on driver ≥ 550)
# cu121: CUDA 12.1  (3090, A100, 4090 on older drivers)
# cu118: CUDA 11.8  (older drivers; only if 12.x isn't available)
python3.11 -m pip uninstall -y torch
python3.11 -m pip install --upgrade torch \
  --index-url https://download.pytorch.org/whl/cu124
python3.11 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA GeForce RTX 4090   (or your GPU)
```

If `torch.cuda.is_available()` returns False after reinstall, you have a driver mismatch — check `nvidia-smi` output's CUDA version against the wheel index URL.

### 2. Edit `configs/cisc187_pt.yaml`

Three changes:

```yaml
### train
use_cpu: false               # was true → flip to false (or delete the line)

bf16: true                   # was false → enable BF16. All Ampere+ GPUs support it natively.
fp16: false                  # leave false; bf16 is strictly better when supported.

per_device_train_batch_size: 4   # was 1 → bump because GPU has the VRAM headroom
gradient_accumulation_steps: 2   # was 4 → drop accordingly (effective batch stays the same)
```

For a 24 GB GPU (RTX 4090) training a 7B model, you can sometimes push `per_device_train_batch_size: 8` with `gradient_accumulation_steps: 1`. Monitor `nvidia-smi` during the first 100 steps — if you're below 80% VRAM usage, bump batch size; if you're hitting OOM, drop it.

### 3. (Optional) Use QLoRA to fit bigger models

For 7B+ on a 24 GB consumer GPU, QLoRA (4-bit quantized base + LoRA on top) is the way to go. Add to the YAML:

```yaml
### method
quantization_bit: 4              # 4-bit NF4 (default in LLaMA-Factory)
quantization_method: bitsandbytes
```

This drops VRAM by ~4× during training. The merged output is still FP16 quality — you only lose precision during gradient computation. Documented to work cleanly for 7B/13B on a 24 GB card ([craftrigs](https://craftrigs.com/guides/fine-tuning-7b-llm-consumer-gpu-unsloth-lora/)).

### VRAM math per model size

| Model    | Full LoRA (BF16) | QLoRA (INT4) | Minimum GPU                         |
|----------|------------------|--------------|-------------------------------------|
| 1B       | ~6 GB            | ~3 GB        | Any 8 GB+ GPU                       |
| 3B       | ~12 GB           | ~6 GB        | RTX 3060 12GB / 4060 16GB           |
| 8B       | ~24 GB           | ~10 GB       | RTX 4090 24GB (LoRA fits at FP16)   |
| 13B      | ~32 GB           | ~16 GB       | RTX 5090 32GB (LoRA fits at FP16)   |
| 70B      | ~320 GB          | ~48 GB       | QLoRA: A100 80GB or H100 80GB       |

Sources: [llmhardware.io](https://llmhardware.io/guides/llm-fine-tuning-hardware-requirements), [DigitalOcean](https://www.digitalocean.com/resources/articles/gpu-options-finetuning).

### Single GPU vs. multi-GPU

- **Single GPU is fine for ≤ 13B with LoRA, or ≤ 70B with QLoRA on an 80GB card.** No code changes needed.
- **Multi-GPU is needed for full-precision 70B+ fine-tunes.** LLaMA-Factory supports both DeepSpeed ZeRO-3 and FSDP — add `deepspeed: configs/ds_z3_config.json` or `fsdp: full_shard auto_wrap` to the YAML. This is well outside the POC scope; consult LLaMA-Factory's `examples/` directory.

### Expected speedup

| Config                    | Llama-3.2-1B, 5 epochs | Llama-3.2-3B, 3 epochs |
|---------------------------|------------------------|------------------------|
| 2019 i9 CPU (this project)| ~10 h                  | ~20 h                  |
| RTX 3060 12 GB            | ~30 min                | ~2 h                   |
| RTX 4090 24 GB            | ~10 min                | ~40 min                |
| H100 80 GB                | ~3 min                 | ~15 min                |

Rough numbers — actual speedup depends on `cutoff_len`, batch size, and how memory-bound vs compute-bound your config is.

---

## Training parameters explained — for someone who's never fine-tuned before

This section walks through every knob in `configs/cisc187_pt.yaml` so a newcomer can decide what to tweak (and why) without breaking things. The defaults are reasonable for the 1B POC; bullets below tell you what each parameter actually does and when to change it.

### The LoRA knobs

LoRA ("Low-Rank Adaptation") is the technique that makes fine-tuning a big model on consumer hardware feasible. Instead of updating all of the model's weights (which would cost hundreds of GB for 70B), you freeze the original weights and train two small "adapter" matrices that get added on top. The size of those adapters is controlled by `lora_rank`.

```yaml
finetuning_type: lora    # keep this — full fine-tuning would 10× the RAM cost
lora_target: all         # apply LoRA to every linear layer in the model
lora_rank: 32            # adapter capacity; the "how much can it learn" knob
lora_alpha: 64           # scaling factor; convention is alpha = 2 × rank
lora_dropout: 0.05       # regularization; prevents overfit on small corpora
```

- **`lora_rank`** controls how much the adapter can learn. Higher rank = more capacity to memorize specifics, but more RAM and slower training. 8–16 is "light touch" (style transfer, instruction adaptation), 32–64 is "memorize the corpus" (what we want), 128+ is "approaches full fine-tune capacity" (rarely needed).
  - **Bump to 64** if your corpus is large (10M+ tokens) and the loss flattens early.
  - **Drop to 16** if you only have 8 GB free RAM and training keeps OOM'ing.
- **`lora_alpha`** scales how much the adapter influences output. Keep it at 2× rank; this is convention and almost no one tweaks it.
- **`lora_dropout`** randomly zeros out 5% of the adapter during each training step, acting as a regularizer. Keep at 0.05 for textbook-style corpora; bump to 0.1 if you see signs of overfitting (training loss drops but eval answers get worse).

### Training schedule knobs

```yaml
num_train_epochs: 5.0           # how many passes over the corpus
learning_rate: 2.0e-4           # how big each weight update is
lr_scheduler_type: cosine       # how the learning rate changes over time
warmup_ratio: 0.03              # first 3% of steps are at a low LR (stability)
```

- **`num_train_epochs`** — the single biggest knob for memorization quality. More epochs = the model sees each chunk of your corpus more times = better recall, up to a point. After convergence, more epochs cause overfitting (the model memorizes training samples verbatim but loses general reasoning). Read `trainer_log.jsonl` to know when to stop (see below).
  - Default 5.0 for 1B is empirically what landed `train_loss ≈ 1.3` in the project's POC run.
  - For 3B drop to 3.0 (3B converges faster because it has more capacity).
  - For 8B or 70B drop to 2.0.
  - For a much bigger corpus (the recipient's actual C++ codebase), keep 5–7 epochs and watch the loss curve.
- **`learning_rate`** — how aggressively each gradient step updates the LoRA weights. `2e-4` is the LLaMA-Factory recommended default for LoRA continued pretraining. Lower (`1e-4`) for very small corpora to avoid overfitting; higher (`3e-4`) for larger corpora where you want faster convergence.
- **`lr_scheduler_type: cosine`** — starts at `learning_rate`, smoothly decays to ~0 by the end of training. Other options: `linear` (also decays, more aggressively), `constant` (no decay; only use if you're stopping training manually). Cosine is the standard.
- **`warmup_ratio`** — for the first 3% of training, the learning rate ramps from 0 up to `learning_rate`. This prevents the optimizer from making huge updates before it's seen any data. Keep at 0.03 unless training crashes in the first few steps (then bump to 0.06).

### Dataset and packing knobs

```yaml
dataset: cisc187_pt              # the entry in dataset_info.json to load
cutoff_len: 2048                 # max sequence length the model sees per step
max_samples: 100000              # safety cap; lower to truncate a huge corpus
preprocessing_num_workers: 4     # parallel data-prep workers (CPU only)
packing: true                    # concatenate short files into full sequences
```

- **`cutoff_len`** — how many tokens of context the model sees per training step. Bigger = more context, but RAM grows linearly with this value and CPU training slows down. 2048 is a sensible default; bump to 4096 if you have RAM headroom and your corpus has long files (say, single C++ files with 1000+ lines).
- **`packing: true`** — without packing, every short file gets padded with `<pad>` tokens to fill `cutoff_len`, wasting compute on padding. With packing, the loader concatenates multiple short files (separated by EOS) into a single full-length sequence. **Huge throughput win for code corpora** — keep it on.
- **`preprocessing_num_workers`** — how many parallel workers tokenize your data before training starts. 4 is fine for most machines; bump to your physical core count if data prep takes a long time at the start of training.

### Mixed precision

```yaml
bf16: false                # BF16 — only enable on AMX-capable CPUs or any modern GPU
fp16: false                # FP16 — older mixed-precision mode; less stable than BF16
```

- **CPU**: leave both `false` (FP32) unless your CPU has AMX (Intel Sapphire Rapids+, EPYC Genoa+). Check with `grep -o 'amx[_a-z0-9]*' /proc/cpuinfo | sort -u`.
- **GPU**: flip `bf16: true`. Every Ampere+ GPU (RTX 30/40/50 series, A100, H100) supports BF16 natively and it cuts memory by ~50% with no quality loss.

### How to read `trainer_log.jsonl` to know when to stop or tweak

After training starts, watch:

```bash
tail -f ~/llf-bundle/output/llama-3.2-1b-cisc187-lora/trainer_log.jsonl
```

Each line is a JSON record like:

```json
{"current_steps": 10, "total_steps": 200, "loss": 2.4153, "learning_rate": 6.6e-05, "epoch": 0.25}
```

Healthy progression looks like:

| Epoch | Expected `loss` for 1B + cisc187 corpus |
|-------|-----------------------------------------|
| 0.5   | ~2.4                                    |
| 1.0   | ~2.0                                    |
| 2.0   | ~1.7                                    |
| 3.0   | ~1.5                                    |
| 5.0   | ~1.3 (converged)                        |

**Signs of trouble:**

- **Loss is flat at 2.5+** after the first few steps → the data pipeline is broken. Usually means `dataset_info.json`'s `file_name` path is wrong, or the JSONL has the wrong column key. Kill the run and re-check Step 4d.
- **Loss drops fast then climbs** ("loss spike") → learning rate is too high. Drop `learning_rate` to `1.0e-4` and restart.
- **Loss drops to ~0.5 then keeps dropping** → overfitting. The model is memorizing training samples verbatim. Drop epochs or add more `lora_dropout`.
- **Loss is still steadily dropping at the configured final epoch** → you stopped too early. Bump `num_train_epochs` by 2 and restart.

### What to actually tweak for your corpus

When the recipient swaps in their own C++ codebase, the things most likely to need adjustment:

1. **`cutoff_len: 2048` → `4096`** if individual C++ files frequently exceed 2K tokens (most production code has long files).
2. **`num_train_epochs`** based on corpus size:
   - Small (< 200K tokens): 5–7 epochs
   - Medium (200K–2M tokens): 3–5 epochs
   - Large (> 2M tokens): 2–3 epochs
3. **`max_samples: 100000`** if the corpus exceeds 100K JSONL records (the textbook had 205, way under cap; a real codebase might have 5000+). Either bump this cap or accept truncation.
4. **`lora_rank: 32` → `64`** for codebases bigger than ~1M tokens — bigger adapter to memorize more identifiers.

Everything else should stay at the defaults unless training crashes or produces clear pathologies.

---

## Adapting to your own C++ codebase

The cisc187 textbook is a **deliberate "hello world"** for this pipeline — it's small (~205 files, ~150K tokens), self-contained, and mixes prose with code so the model has something to talk about. The intended real use case is fine-tuning on an actual C++ project so the model "knows" the codebase's identifiers, idioms, and patterns.

### Step 1 — Put your repo on disk

Drop your repo at `~/llf-bundle/datasets/<repo-name>/` (or anywhere; the path is configurable in the next step). For an air-gapped target, transfer it the same way as the textbook — USB or scp from the networked machine.

```bash
cp -r ~/path/to/your-cpp-repo ~/llf-bundle/datasets/your-cpp-repo
```

### Step 2 — Run the data-prep script

`scripts/prepare_cisc187.py` is corpus-agnostic. Despite the name, it already handles C++ source files:

```python
# scripts/prepare_cisc187.py, lines ~26–32
INCLUDE_EXTS = {
    ".rst", ".txt", ".md",
    ".py",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".cmake", ".yml", ".yaml", ".json",
    ".sh",
}
```

If your repo uses other extensions worth training on (e.g. `.tpp` template implementation files, `.inl`, `.proto`, `.thrift`), add them to that set. Conversely, prune extensions you don't want (e.g. drop `.json` if your repo has lots of test fixtures).

Run the script pointing at your repo:

```bash
python3.11 scripts/prepare_cisc187.py \
  --repo ~/llf-bundle/datasets/your-cpp-repo \
  --out  ~/llf-bundle/datasets/your-cpp-repo_pt.jsonl
```

The script will print how many records it wrote and how many it skipped (binary files, empty files). Sanity-check the count is roughly proportional to your repo size — if you get 0 records, something is wrong (likely all source files are in a directory that matches `SKIP_DIR_NAMES`).

### Step 3 — Decide: replace cisc187_pt, or add a new dataset

Two paths, both fine:

**Path A — Reuse the `cisc187_pt` dataset name.** Edit `~/llf-bundle/LLaMA-Factory/data/dataset_info.json` and update the `cisc187_pt` entry's `file_name` to point at your new JSONL:

```json
"cisc187_pt": {
  "file_name": "/Users/<you>/llf-bundle/datasets/your-cpp-repo_pt.jsonl",
  "columns": { "prompt": "text" }
}
```

No other config changes needed. The downside is the name "cisc187_pt" is now misleading; the upside is one fewer thing to edit.

**Path B — Add a new dataset entry.** Add a new block alongside the existing `cisc187_pt`:

```json
"your_cpp_repo_pt": {
  "file_name": "/Users/<you>/llf-bundle/datasets/your-cpp-repo_pt.jsonl",
  "columns": { "prompt": "text" }
}
```

Then edit `configs/cisc187_pt.yaml`:

```yaml
dataset: your_cpp_repo_pt     # was: cisc187_pt
```

This keeps the textbook setup intact for reference, and your new model is named after its actual source. Recommended.

### Step 4 — Update output paths

In `configs/cisc187_pt.yaml`, change `output_dir` to a name that reflects your model:

```yaml
output_dir: /Users/<you>/llf-bundle/output/llama-3.2-1b-yourrepo-lora
```

And rename the Modelfile + update its FROM path:

```bash
cp modelfiles/Modelfile.llama-3.2-1b-cisc187 modelfiles/Modelfile.llama-3.2-1b-yourrepo
# Edit the new file:
# - FROM /Users/<you>/llama-3.2-1b-yourrepo-f16.gguf
# - Update all `llama-3.2-1b-cisc187` tag references inside to `llama-3.2-1b-yourrepo`
# - Update the SYSTEM prompt to describe your repo, not the cisc187 textbook
```

### Step 5 — Tune training params for your corpus size

Before launching, check the JSONL record count and rough token count:

```bash
wc -l ~/llf-bundle/datasets/your-cpp-repo_pt.jsonl
# Optional: rough token count (1 token ≈ 4 chars for code)
du -sb ~/llf-bundle/datasets/your-cpp-repo_pt.jsonl
```

Then per the "Training parameters explained" section above:

- < 200K tokens → keep epochs at 5
- 200K–2M tokens → drop epochs to 3, consider `cutoff_len: 4096`
- \> 2M tokens → drop epochs to 2, definitely bump `cutoff_len`, consider `lora_rank: 64`, raise `max_samples` if it would truncate

### Step 6 — Train, merge, import, run

Run Step 4f's training command unchanged. The merge/convert/import in Step 4g works identically — just substitute your new output dir name and Modelfile name throughout. Once `ollama run llama-3.2-1b-yourrepo "where is class Foo defined?"` answers coherently, you're done.

### What you'll see (and what you won't)

Fine-tuning teaches the model **patterns and identifiers** from your repo. What it does well:

- Knows what files contain what (e.g. `Foo` lives in `src/foo/`).
- Reproduces your code style (naming conventions, error handling patterns).
- Recognizes domain-specific types (`UserId`, `RequestContext`, etc.).

What it doesn't do:

- Won't reliably reproduce **specific function bodies** — that's verbatim recall, which a 1B model lacks the capacity for. Scale to 3B+ for that.
- Won't know about code changes since the last training run. Fine-tuning is a snapshot; re-train periodically as the repo evolves.

For both of those limitations, the Goose agent (Step 3) is a complementary tool — it can `read_file` to get exact current source on demand, where the fine-tune gives it the cross-cutting "shape" of the codebase.

