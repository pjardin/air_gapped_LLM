# air_gapped_LLM — quick reference

Copy-paste checklist for the air-gapped fine-tune pipeline. **No explanations here** — for why each step exists, tradeoffs, platform caveats, and recovery when things go wrong, see **[README_DETAILED.md](./README_DETAILED.md)**.

Pipeline: install Python 3.11 → install Ollama → download base model → install Goose (CLI AI coding agent with filesystem tools) → fine-tune with LLaMA-Factory → run the merged model from Ollama.

Two machines assumed: a **networked** machine to download everything, and the **air-gapped target** (mid-tier CPU server, 16–64 cores, no GPU) to run on. Transfer between them by USB / scp / internal mirror.

**Where to put things on the target machine** — every command below assumes both of these directories live directly under your home folder:

```
~/air_gapped_llm/     # this repo (clone or extract here)
~/llf-bundle/         # the LLaMA-Factory bundle assembled in Step 4a
```

Place them side-by-side in `$HOME` so relative paths in the configs and scripts resolve consistently. If you put them somewhere else, you'll have to edit the `~/llf-bundle/...` paths in `configs/cisc187_pt.yaml`, the Modelfiles, and Step 4f's `cd ~/air_gapped_llm`.

---

## Hardware requirements — read this before starting

This whole pipeline is **CPU-only by default** (no GPU). Pick the row that matches your hardware; the column tells you which model you can realistically fine-tune in a reasonable time. RAM is the dominant constraint, then CPU core count. For GPU instead, see the "Running on GPU" section below.

| Your machine                            | Free RAM | Physical cores | Fine-tune target          | Wall-clock for full training |
|-----------------------------------------|---------:|---------------:|---------------------------|------------------------------|
| **POC laptop** (16 GB MacBook / desktop) | ~8 GB   | 4–8            | `Llama-3.2-1B` only       | 5 epochs ≈ 6–12 h on a 2019 i9 |
| **Comfort tier** (32–64 GB)              | ~24 GB   | 8–16           | up to `Llama-3.2-3B`      | 3B × 3 epochs ≈ 20 h on a 2019 i9 |
| **Workstation** (128 GB Linux)           | ~96 GB   | 32+            | up to `Llama-3.1-8B`      | 8B × 3 epochs ≈ 1–2 days     |
| **Server** (256 GB+ Linux)               | ~200 GB  | 64+            | up to `Llama-3.3-70B`     | 70B × 3 epochs ≈ multi-day   |

**Inference (running the model via Ollama) is much cheaper than training.** A model that *trains* in 12 h on 8 GB RAM will *answer prompts* in < 1 GB of RAM overhead once quantized to Q4_K_M.

> **For a Mac specifically:** training uses CPU only (Apple's MPS / Metal backend isn't fully supported by LLaMA-Factory's optimizer path). An M-series Mac with 64 GB unified memory is still capable: roughly the same wall-clock as a 2019 Intel i9 for 1B–3B fine-tunes.

### Picking a base model — quality vs. cost

| Model                                  | Quality on cisc187 textbook POC  | Train time on Mac CPU | Best for                                             |
|----------------------------------------|----------------------------------|-----------------------|------------------------------------------------------|
| `unsloth/Llama-3.2-1B-Instruct`        | ★★☆☆☆ — coherent but mediocre   | 3–12 h                | **POC default.** Validates the pipeline end-to-end. |
| `unsloth/Llama-3.2-3B-Instruct`        | ★★★★☆ — production-grade answers | ~20 h                 | The natural production scale-up after POC succeeds. |
| `unsloth/Meta-Llama-3.1-8B-Instruct`   | ★★★★★ — strong recall + framing  | ~18 h/epoch (Mac); 6–8 h/epoch on 32-core Linux | Real production runs; Linux preferred.            |
| `unsloth/Llama-3.3-70B-Instruct`       | ★★★★★+ — best open-weights quality | Multi-day on Linux  | Server-class; needs ≥96 GB RAM and a multi-day window. |

Star rating reflects fine-tune Q&A quality on a textbook-style corpus like cisc187 — your mileage may vary on a different domain. **Recommendation: start at 1B for the POC even if you have hardware for 3B, because the order-of-magnitude faster iteration loop catches pipeline bugs cheap.**

### Running on GPU instead of CPU

Single GPU drops training time from days to hours. The pipeline can be swapped to GPU with three changes:

1. **Reinstall PyTorch with CUDA wheels** instead of the CPU index:
   ```bash
   python3.11 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
   ```
   (Use `cu121` for Ampere/older GPUs that don't support CUDA 12.4.)

2. **Edit `configs/cisc187_pt.yaml`:**
   - Remove or set `use_cpu: false`
   - Set `bf16: true` (any Ampere+ GPU supports BF16 natively)
   - Bump `per_device_train_batch_size` from 1 → 4 or 8 (depends on VRAM)
   - Drop `gradient_accumulation_steps` accordingly (e.g. 4 → 1)

3. **GPU count needed by model size** (with LoRA + BF16, no quantization):

   | Model        | Minimum VRAM | Realistic single-GPU | Multi-GPU notes                                       |
   |--------------|--------------|----------------------|-------------------------------------------------------|
   | 1B           | 8 GB         | RTX 3060 12 GB / 4060| 1 GPU is overkill — CPU is fine for this size        |
   | 3B           | 16 GB        | RTX 4080 16 GB       | 1 GPU                                                |
   | 8B           | 24 GB        | RTX 4090 / 5090      | 1 GPU (QLoRA fits comfortably at INT4)               |
   | 70B (QLoRA)  | 48 GB        | A100 80GB / H100     | 1× 80 GB GPU minimum for QLoRA; 2× for room          |
   | 70B (LoRA)   | 320 GB+      | n/a                  | Needs 4× A100 80GB or 4× H100 with FSDP/DeepSpeed   |

   For deep numbers, GPU-vs-CPU speedups, and the full set of YAML changes, see README_DETAILED.md → "Running on GPU — full walkthrough."

### Using your own repo (not the cisc187 textbook)

The pipeline is corpus-agnostic. To train on a real C++ codebase instead of the included textbook:

1. **Drop your repo at** `repo_to_fine_tune/<your-repo-name>/`.
2. **Run the prep script** (it already handles `.cpp`, `.cc`, `.cxx`, `.c`, `.h`, `.hpp`, `.hxx`, plus CMake/Make/shell/YAML/JSON):
   ```bash
   python3.11 scripts/prepare_cisc187.py \
     --repo ~/llf-bundle/datasets/<your-repo-name> \
     --out  ~/llf-bundle/datasets/<your-repo-name>_pt.jsonl
   ```
3. **Update `~/llf-bundle/LLaMA-Factory/data/dataset_info.json`** so the `cisc187_pt` entry's `file_name` points at the new `.jsonl` (or add a new entry with a different dataset key and update `dataset:` in `configs/cisc187_pt.yaml`).
4. **Train as normal.** The fine-tune will learn your repo's identifiers, idioms, and patterns. Plan ≥1 epoch per ~150K tokens of corpus for the model to memorize meaningfully; larger codebases want more epochs.

The script also drops files under `.git/`, `build/`, `node_modules/`, etc. by default. To include or exclude specific patterns, edit `INCLUDE_EXTS` / `SKIP_DIR_NAMES` at the top of `scripts/prepare_cisc187.py`.

---

## Step 0 — Install Python 3.11 (Anaconda)

This whole pipeline standardizes on **Python 3.11**, installed via **Anaconda**. Anaconda bundles Python 3.11 plus ~250 data-science libraries (NumPy, pandas, SciPy, scikit-learn, Jupyter, etc.) in one offline installer — useful for the air-gapped target where you can't `pip install` extras later.

Pin to **Anaconda 2023.09-0** — the last Anaconda release that bundles Python 3.11 by default. 2024.x and later switched to 3.12.

### Download (on a networked machine)

Anaconda ships both `.pkg` (GUI installer) and `.sh` (shell installer) for macOS. We use the **`.sh`** because it installs into `$HOME/anaconda3` without sudo and avoids macOS's system-volume protection error (`"The package is attempting to install content to the system volume."`) that the `.pkg` hits on modern macOS.

```bash
# macOS (Apple Silicon)
curl -L https://repo.anaconda.com/archive/Anaconda3-2023.09-0-MacOSX-arm64.sh   -o Anaconda3-2023.09-0-MacOSX-arm64.sh

# macOS (Intel)
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
bash Anaconda3-2023.09-0-MacOSX-arm64.sh -b -p $HOME/anaconda3   # or -x86_64.sh on Intel
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
python3.11 --version
```

**Linux:**
```bash
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p $HOME/anaconda3
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
python3.11 --version
```

`-b -p $HOME/anaconda3` = batch mode (no interactive prompts) + install prefix. Same flags for macOS and Linux.

**Windows:** double-click the `.exe`, follow the GUI installer, check "Add Anaconda3 to my PATH environment variable" when prompted. Then open a new Command Prompt and run `python --version`.

### Verify

```bash
python3.11 --version            # expect Python 3.11.x
python3.11 -m pip --version
```

Once `python3.11 --version` resolves on the target, proceed to Step 1.

---

## Step 1 — Install Ollama

**Networked machine — download:**

```bash
curl -L https://ollama.com/download/Ollama.dmg               -o Ollama.dmg                          # macOS
curl -L https://ollama.com/download/OllamaSetup.exe          -o OllamaSetup.exe                     # Windows
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst        # Linux x86_64
curl -L https://ollama.com/download/ollama-linux-arm64.tar.zst -o ollama-linux-arm64.tar.zst        # Linux ARM64
```

**Target — install (Linux example):**

```bash
sudo tar -I zstd -xvf ollama-linux-amd64.tar.zst -C /usr
```

**Target — append to `~/.bashrc` *before* starting Ollama:**

```bash
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NOPRUNE=true
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
```

```bash
source ~/.bashrc
ollama serve &        # or: nohup ollama serve > ~/ollama.log 2>&1 &
ollama --version
ollama ps
```

---

## Step 2 — Pull the base model from Ollama's library

> **What this step does:** put a model into Ollama so you can chat with it AND use it as the brain for the Step 3 Goose agent. The model needs to be **tools-capable** so Goose can give it filesystem tools; we pull from Ollama's official library (not a custom GGUF) because library models have their chat templates configured to advertise tool capability.
>
> **This is NOT the training input.** Fine-tuning needs the HuggingFace safetensors version, downloaded separately in **Step 4a**.

**Networked machine — pull `llama3.1:8b` from Ollama's library:**

```bash
ollama pull llama3.1:8b
ollama show llama3.1:8b | grep -i tools   # "Capabilities" section should include tools
ollama run llama3.1:8b "hello"
```

~5 GB on disk, **~6 GB RAM** to run at Q4_K_M. Meta's dense 8B Llama with native tool-use support — the smallest Llama that's genuinely useful for Goose agent loops, and it fits on basically any modern Mac. One model for both interactive chat AND the Step 3 Goose agent — no custom Modelfile needed because Ollama's library version comes with the right template and `tools` capability.

If you have ≥48 GB free RAM and want stronger multi-step tool reasoning, scale up to `llama3.3:70b` instead — same `ollama pull` flow, also tools-capable, much slower on CPU (~1–3 tok/s vs. ~10–15 for the 8B) but noticeably better at chaining four or more tool calls.

#### Air-gap transfer (when ready to move to the offline target)

Ollama doesn't have a dedicated `export` command, but its on-disk format is just `manifests/` + `blobs/` under `~/.ollama/models/`. To move the pulled model to an air-gapped target:

```bash
# On networked machine (after `ollama pull llama3.1:8b`)
tar czf ollama-llama3.1-8b.tgz -C ~/.ollama models/

# Transfer to air-gapped target, then:
mkdir -p ~/.ollama
tar xzf ollama-llama3.1-8b.tgz -C ~/.ollama/
ollama list      # should now show llama3.1:8b
```

The tarball will be ~5 GB (the blob is the bulk). If you have other Ollama models on the networked machine you don't want to ship, `find ~/.ollama/models/manifests` first to see what's there and tar only the `llama3.1` subtree.

#### Other Ollama-library tags (reference — not pulled by default)

If you want to compare different models for the agent role, anything in Ollama's library tagged as tools-capable will work. Confirm with `ollama show <tag> | grep -i tools` before using (the "Capabilities" section in the default `ollama show` output should list `tools`):

```
llama3.1:8b           # default; dense 8B, tools-capable, ~6 GB RAM
llama3.2:3b           # smaller / faster; tools-capable but weaker at multi-step
llama3.2:1b           # smallest tools-capable Llama; smoke testing only
llama3.1:70b          # bigger / slower / stronger multi-step; ~40 GB RAM
llama3.3:70b          # newest 70B, matches Llama 3.1 405B on tool use; ~45 GB RAM
```

To switch, just `ollama pull <tag>` and update `GOOSE_MODEL` in `configs/goose-config.yaml`.

#### Context window — bump it for real code work

Ollama defaults to a 4K–8K window regardless of what the model architecturally supports. That's far too small for anything beyond chat: one 500-line source file is already ~3K tokens, and an agent loop layers a system prompt + tool schemas + tool outputs on top of it. To do meaningful repo work you need both a **bigger model** (smaller models lose-in-the-middle even when long context fits) and a **bigger window**.

Cheapest one-off: pass `"options": {"num_ctx": 32768}` in any `/api/chat` request. Permanent: write a derived Modelfile with `PARAMETER num_ctx 32768` and `ollama create llama3.1-8b-32k -f thatfile`, then point `GOOSE_MODEL` at the new tag. Global default: `export OLLAMA_CONTEXT_LENGTH=32768` before `ollama serve`.

Doubling `num_ctx` roughly doubles the KV-cache RAM on top of the model weights — `llama3.1:8b` at 32K is ~10 GB, at 128K is ~30 GB. See **README_DETAILED.md → "Context window size — how to change it, and why it's load-bearing for repo work"** for the full table and the "even 128K isn't enough for a real repo" discussion.

---

## Step 3 — Install Goose (AI coding agent with filesystem tools)

[Goose](https://github.com/block/goose) is Block's open-source CLI coding agent — the closest offline analog to Claude Code. Unlike Aider, which just shuttles file contents into the prompt, **Goose gives the model real tools** (`list_files`, `read_file`, `run_shell`, etc.) so the model autonomously explores the repo as it works. Single static binary, no Python dep tangle.

**Networked machine — download the binary for your air-gapped target's OS/arch:**

```bash
# Linux x86_64 (most likely the air-gapped target)
curl -L https://github.com/block/goose/releases/latest/download/goose-x86_64-unknown-linux-gnu.tar.bz2 -o goose-linux-x86_64.tar.bz2

# Linux ARM64
curl -L https://github.com/block/goose/releases/latest/download/goose-aarch64-unknown-linux-gnu.tar.bz2 -o goose-linux-arm64.tar.bz2

# macOS Apple Silicon
curl -L https://github.com/block/goose/releases/latest/download/goose-aarch64-apple-darwin.tar.bz2 -o goose-macos-arm64.tar.bz2

# macOS Intel
curl -L https://github.com/block/goose/releases/latest/download/goose-x86_64-apple-darwin.tar.bz2 -o goose-macos-x86_64.tar.bz2
```

Verify the asset name on the GitHub releases page if a curl 404s — Block occasionally renames assets between releases.

**Target:**

```bash
# Extract into a directory on PATH (use ~/bin on macOS, /usr/local/bin on Linux)
mkdir -p ~/bin
tar xjf goose-linux-x86_64.tar.bz2 -C ~/bin/        # or goose-macos-x86_64.tar.bz2
chmod +x ~/bin/goose

# Put ~/bin on PATH for the shell
RC=~/.bashrc; [ "$(uname)" = "Darwin" ] && RC=~/.zshrc
echo 'export PATH="$HOME/bin:$PATH"' >> "$RC"
source "$RC"

goose --version

# Drop in the Ollama+Llama config (or run `goose configure` interactively).
mkdir -p ~/.config/goose
cp configs/goose-config.yaml ~/.config/goose/config.yaml

# Start an interactive session in any repo to test
cd ~/Documents/GitHub/air_gapped_LLM
goose session
# Inside the session, try: "what's in the repo_to_fine_tune directory?"
```

The agent should autonomously call `list_files` / `read_file` to answer — no `/add` ceremony like Aider required.

### Step 3b — Add skills and extensions (plugins) to the agent

Goose has **two** ways to extend the agent, and they do different jobs:

| Mechanism | Gives the agent… | Is a… | Lives in |
|-----------|------------------|-------|----------|
| **Extension (MCP server)** | new **tools/abilities** (e.g. query a DB, hit an internal API, drive git) | local process Goose talks to over stdio | `extensions:` block in `config.yaml` |
| **Skill (`SKILL.md`)** | new **know-how** (procedural instructions for a task) | folder of markdown + optional helper files | `~/.config/agents/skills/<name>/` or `.goose/` in the repo |

Block's own one-liner: *MCP gives agents abilities; skills teach agents how to use those abilities well.* Use an extension when the agent needs to *do* something new; use a skill when it already has the tools but needs a *playbook*.

> **Air-gap rule for both:** nothing may phone home at runtime. The common `npx -y <server>` / `uvx <server>` examples in upstream Goose docs **download the server on first run** — that fails on the offline target. Pre-stage every server binary/script (and any skill files) on the networked machine, ship them over with the rest of the bundle, and point Goose at the **local path**.

#### A) Add an extension (MCP server / "plugin")

Two ways — interactive or by editing the config directly.

**Interactive:**

```bash
goose configure
# → Add Extension → Command-line Extension (runs a local STDIO MCP server)
#   Name:    my-tool
#   Command: python3.11 /home/<you>/goose-extensions/my_server.py
#   Timeout: 300
```

**Or edit `~/.config/goose/config.yaml`** — add a `type: stdio` entry alongside the bundled ones already in `configs/goose-config.yaml`:

```yaml
extensions:
  my_tool:
    enabled: true
    type: stdio                 # local process over stdio (not sse/streamable_http — those need the network)
    name: my_tool
    description: What this tool does, in one line (the model reads this to decide when to call it)
    cmd: python3.11             # a binary/interpreter that already exists on the target
    args:
      - /home/<you>/goose-extensions/my_server.py   # absolute path to the vendored MCP server
    envs: {}                    # e.g. { DB_PATH: /home/you/data/app.db }
    timeout: 300
    bundled: false
```

Restart any open `goose session` to pick it up, then confirm the new tools registered:

```bash
goose info -v        # lists active extensions and their tools
```

Keep the menu small. As noted in `configs/goose-config.yaml`, a local 8B model gets measurably worse at picking the right tool when the tool list is long — add only the extensions a given task actually needs and leave the rest `enabled: false`.

#### B) Add a skill (`SKILL.md`)

First enable the bundled **skills** extension (it ships `enabled: false` in `configs/goose-config.yaml`):

```yaml
extensions:
  skills:
    enabled: true       # flip from false
    type: platform
    name: skills
    description: Discover and provide skill instructions from filesystem and builtins
    display_name: Skills
    bundled: true
    available_tools: []
```

Then drop a skill folder where Goose discovers them — globally for every session, or per-repo:

```bash
# Global: available in any goose session on this machine
mkdir -p ~/.config/agents/skills/cpp-review
$EDITOR ~/.config/agents/skills/cpp-review/SKILL.md

# Per-repo: ships with the project, only loads inside this repo
mkdir -p ~/air_gapped_llm/.goose/skills/cpp-review
```

A `SKILL.md` is YAML frontmatter (`name` + `description` are the required fields) plus a markdown body of instructions:

```markdown
---
name: cpp-review
description: Review C++ for the cisc187 textbook style — checks headers, naming, and RAII. Use when asked to review or critique C++ in this repo.
---

When reviewing C++ in this repo:
1. Read the file with the developer tool before commenting.
2. Flag raw `new`/`delete` and suggest smart pointers.
3. Check headers use `#pragma once`.
4. Keep feedback to a short bulleted list.
```

Goose loads the skill on demand: when a task matches the `description`, it pulls the body into context and follows it. The folder can also hold helper scripts or templates the skill references. Skills are just files, so they transfer over the air gap with `scp`/USB like everything else — no install step that touches the network.

---

## Step 4 — Fine-tune

### 4a. Networked machine — gather everything

**Pick a base model to fine-tune.** Llama-3.2-1B-Instruct is the **current default** — the walkthrough below is set up for it end-to-end (YAML, Modelfile, eval). It fine-tunes in ~3–6 h on a Mac CPU, which is the right cost for the POC. Llama-3.2-3B-Instruct is the production scale-up: noticeably better Q&A quality but ~20 h on the same Mac.

| Size class                         | Recommended HF path                      | Notes                                                                   |
|------------------------------------|------------------------------------------|-------------------------------------------------------------------------|
| **Default (Mac POC)**              | `unsloth/Llama-3.2-1B-Instruct`          | **The walkthrough below uses this.** ~1–2 h/epoch on Mac CPU, 5 epochs ≈ ~3–6 h. Pipeline validation; coherent-but-mediocre answers. |
| Production (Mac)                   | `unsloth/Llama-3.2-3B-Instruct`          | Scale-up after POC. ~4–5 h/epoch, 3 epochs ≈ ~20 h. Real Q&A quality. |
| Bigger (Linux target)              | `unsloth/Meta-Llama-3.1-8B-Instruct`     | ~18 h/epoch on Mac (impractical). On a 32+ core Linux box, ~6–8 h/epoch is fine. Production-grade quality. |
| Server-class only                  | `unsloth/Llama-3.3-70B-Instruct`         | Needs ≥96 GB RAM (FP16 + LoRA optimizer state) and multi-day patience even on Linux. Skip on a laptop; only attempt on a real server. |

> **Why `unsloth/...` paths?** Meta's official `meta-llama/...` repos are gated — `hf download` returns `401 GatedRepoError` until you (a) request access on each model's HF page, (b) accept the Llama Community License, and (c) `hf auth login` with a read token. The `unsloth/...` namespace re-uploads the same safetensors verbatim and is ungated, so a fresh machine can `hf download` immediately. For an air-gapped POC the unsloth mirror is the path of least resistance; for a strict reproducibility audit, use the meta-llama paths and pay the auth tax.

The bash below downloads `Llama-3.2-1B-Instruct` (the current POC YAML setting). To use a different size, substitute the HF path in the `hf download` line and update `model_name_or_path` in `configs/cisc187_pt.yaml` to match.

```bash
mkdir llf-bundle && cd llf-bundle

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

mkdir llf-wheels
# Core deps — works on every target.
# [metrics] extra is intentionally omitted: it pulls jieba (sdist-only on PyPI),
# which conflicts with --only-binary=:all:. We don't use BLEU/ROUGE in this
# pipeline — quality is judged by manual inspection of `ollama run` output.
# To put it back later, add --no-binary=jieba alongside --only-binary=:all:.
python3.11 -m pip download \
  --dest ./llf-wheels \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "llamafactory[torch]" "setuptools<81"
# Letting pip pick the torch version (no explicit pin) avoids platform
# resolution failures — pip selects the latest torch compatible with the
# host. On Linux x86_64 that's a recent 2.x with all the APIs LLaMA-Factory
# uses; on older macOS/Intel hosts pip drops to whatever still publishes
# wheels for that platform.

# Optional: Intel CPU accelerator. LINUX x86_64 ONLY — Intel doesn't publish
# macOS or ARM wheels. If the networked machine is a Mac, you'll need to use
# the cross-platform Docker pattern (see README_DETAILED Step 3) or pass
# --platform manylinux2014_x86_64 --abi cp311 --python-version 311 to get
# Linux IPEX wheels.
python3.11 -m pip download \
  --dest ./llf-wheels \
  --only-binary=:all: \
  intel_extension_for_pytorch || echo "skipping IPEX — not available for this platform"

# HuggingFace safetensors (FP16) — separate file from the Step 2 GGUF.
# GGUF is inference-only (can't be loaded back into PyTorch for training);
# fine-tuning needs the safetensors weights + tokenizer files this command
# pulls down (~2.5 GB total for 1B; ~6 GB for 3B; ~16 GB for 8B; ~140 GB for 70B).
python3.11 -m pip install --user huggingface_hub
hf download unsloth/Llama-3.2-1B-Instruct \
  --local-dir Llama-3.2-1B-Instruct

cd ..
tar czf llf-bundle.tgz llf-bundle/
```

### 4b. Target — install

```bash
# Extract into your HOME directory so the paths used in Steps 4d–4g
# (~/llf-bundle/datasets, ~/llf-bundle/LLaMA-Factory, ~/llf-bundle/llama.cpp, etc.)
# resolve correctly. Adjust the tgz path to wherever you transferred it.
cd ~
tar xzf ~/Downloads/llf-bundle.tgz       # results in ~/llf-bundle/
cd ~/llf-bundle

# 1. Core install — works on Linux, macOS, Windows.
python3.11 -m pip install --user --no-index --find-links ./llf-wheels "llamafactory[torch]"

# 2. Pin setuptools to <81 in user-site so pkg_resources is available.
# (Setuptools 81+ removed pkg_resources, which librosa — eagerly imported
# by llamafactory's mm_plugin.py — still uses. --force-reinstall ensures
# user-site gets it even if the system already has setuptools 82.)
python3.11 -m pip install --user --force-reinstall --no-index --find-links ./llf-wheels "setuptools<81"

# 3. Put the user-site CLI scripts on PATH so `llamafactory-cli` is callable
# from any shell. Safe to re-run if Step 3 already added it.
export PATH="$(python3.11 -m site --user-base)/bin:$PATH"
echo 'export PATH="$(python3.11 -m site --user-base)/bin:$PATH"' >> ~/.zshrc    # macOS
# echo 'export PATH="$(python3.11 -m site --user-base)/bin:$PATH"' >> ~/.bashrc # Linux

# 4. Optional: Intel CPU acceleration. LINUX x86_64 + Intel CPU only.
# Skip this line on Mac, ARM, or AMD. If you try it on Mac you'll see
# "ERROR: Could not find a version that satisfies the requirement
# intel_extension_for_pytorch" — that's expected and harmless.
python3.11 -m pip install --user --no-index --find-links ./llf-wheels intel_extension_for_pytorch

llamafactory-cli version
```

**If `llamafactory-cli version` still fails with `ModuleNotFoundError: No module named 'pkg_resources'`** even after step 2 above, the system's setuptools is winning the import path. Uninstall it first, then reinstall the older version into user-site:

```bash
python3.11 -m pip uninstall -y setuptools
python3.11 -m pip install --user --no-index --find-links ./llf-wheels "setuptools<81"
```

### 4c. CPU tuning — append to `~/.bashrc` (replace `32` with your physical core count from `lscpu`)

```bash
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
```

```bash
source ~/.bashrc
lscpu | grep -E "^Socket|^Core|^Thread"                  # verify physical core count
grep -o 'amx[_a-z0-9]*' /proc/cpuinfo | sort -u          # if AMX present, set bf16: true in YAML
```

### 4d. Prepare dataset

```bash
mkdir -p ~/llf-bundle/datasets
cp -r repo_to_fine_tune/cisc187-reader-master ~/llf-bundle/datasets/
python3.11 scripts/prepare_cisc187.py \
  --repo ~/llf-bundle/datasets/cisc187-reader-master \
  --out  ~/llf-bundle/datasets/cisc187_pt.jsonl
```

Edit `LLaMA-Factory/data/dataset_info.json` — add the entry from `configs/dataset_info_patch.json`, replacing the `file_name` placeholder with the absolute path to `cisc187_pt.jsonl`.

### 4e. Edit training config

Open `configs/cisc187_pt.yaml` and replace **3 placeholder paths** with absolute paths for your machine:

| Line  | Field                | Replace with                                                    |
|-------|----------------------|-----------------------------------------------------------------|
| ~9    | `model_name_or_path` | `/Users/<you>/llf-bundle/Llama-3.2-1B-Instruct` (macOS)<br>`/home/<you>/llf-bundle/Llama-3.2-1B-Instruct` (Linux) |
| ~27   | `dataset_dir`        | `/Users/<you>/llf-bundle/LLaMA-Factory/data` (macOS)<br>`/home/<you>/llf-bundle/LLaMA-Factory/data` (Linux) |
| ~34   | `output_dir`         | `/Users/<you>/llf-bundle/output/llama-3.2-1b-cisc187-lora` (macOS)<br>`/home/<you>/llf-bundle/output/llama-3.2-1b-cisc187-lora` (Linux) |

All three must be **absolute** (start with `/`). LLaMA-Factory does not expand `~` or resolve relative paths in these fields — a literal `~/...` or `./...` will be passed straight to `open()` and fail with `FileNotFoundError`. The `output_dir` in particular needs to be absolute because Step 4g's export command looks for the trained adapter there; if `output_dir` is relative, the adapter lands wherever you ran training from and Step 4g can't find it.

Also edit **`configs/dataset_info_patch.json`** (1 line — the `file_name` value) and **merge the entry into `~/llf-bundle/LLaMA-Factory/data/dataset_info.json`** before training:

| File                                 | Line | Field        | Replace with                                                |
|--------------------------------------|------|--------------|-------------------------------------------------------------|
| `configs/dataset_info_patch.json`    | ~3   | `file_name`  | `/Users/<you>/llf-bundle/datasets/cisc187_pt.jsonl` (macOS) |

Once filled in, paste the `cisc187_pt` block from `dataset_info_patch.json` into `~/llf-bundle/LLaMA-Factory/data/dataset_info.json` alongside the existing entries (Step 4d covers this).

### 4f. Train

Training is launched from the **`air_gapped_llm` repo root** (which should live at `~/air_gapped_llm/`) via the `scripts/run_train.py` wrapper — **not** by calling `llamafactory-cli train` directly. The wrapper applies a tiny shim that adds `torch.mps.device_count` if the installed torch is missing it (older versions, e.g. torch 2.2.2 which is the max on macOS Intel). On systems with a recent torch (Linux, Apple Silicon) the shim is a no-op, so the same command works everywhere.

The YAML sets `dataset_dir: ~/llf-bundle/LLaMA-Factory/data` explicitly, so you don't need to `cd` into the LLaMA-Factory directory to launch training — just stay in the repo.

```bash
pkill ollama
cd ~/air_gapped_llm

# Pick one:
python3.11 scripts/run_train.py configs/cisc187_pt.yaml                              # generic
ipexrun python3.11 scripts/run_train.py configs/cisc187_pt.yaml                      # Intel + IPEX
numactl --cpunodebind=0 --membind=0 python3.11 scripts/run_train.py configs/cisc187_pt.yaml  # multi-socket
```

Monitor:

```bash
tail -f output/llama-3.2-1b-cisc187-lora/trainer_log.jsonl
```

### 4g. Merge → convert → import

```bash
cd ~/llf-bundle/LLaMA-Factory
# Use ABSOLUTE paths for all three location flags — `~/...` and relative paths
# are passed straight to open() without expansion (same issue as the YAML).
llamafactory-cli export \
  --model_name_or_path /Users/pascaljardin/llf-bundle/Llama-3.2-1B-Instruct \
  --adapter_name_or_path /Users/pascaljardin/llf-bundle/output/llama-3.2-1b-cisc187-lora \
  --export_dir /Users/pascaljardin/llf-bundle/output/llama-3.2-1b-cisc187-merged \
  --export_size 2 \
  --export_legacy_format false

cd ~/llf-bundle/llama.cpp
python3.11 convert_hf_to_gguf.py \
  /Users/pascaljardin/llf-bundle/output/llama-3.2-1b-cisc187-merged \
  --outfile /Users/pascaljardin/llama-3.2-1b-cisc187-f16.gguf \
  --outtype f16

ollama serve &
# Edit modelfiles/Modelfile.llama-3.2-1b-cisc187 — change FROM to point at
# ~/llama-3.2-1b-cisc187-f16.gguf (the merged model produced just above).
ollama create llama-3.2-1b-cisc187 -f modelfiles/Modelfile.llama-3.2-1b-cisc187
ollama run llama-3.2-1b-cisc187 "what is a binary search tree"
```

---

## Daily Ollama commands

```bash
ollama ps                       # what's running / loaded in RAM
ollama list                     # what's on disk
ollama show <tag>               # model details
ollama show <tag> --modelfile   # exact Modelfile used
ollama rm <tag>                 # delete
ollama stop <tag>               # unload from RAM (keep on disk)
pkill ollama                    # stop daemon (started manually)
```

---

## Repo layout

```
modelfiles/   # Ollama Modelfiles — edit FROM paths before `ollama create`
configs/      # LLaMA-Factory YAML, Goose config, dataset_info patch
scripts/      # prepare_cisc187.py (data prep), run_train.py (training wrapper)
repo_to_fine_tune/cisc187-reader-master/    # POC corpus (C++ textbook in reST)
```

For everything else — explanations, tradeoffs, why a value is what it is, what to do when something breaks — see **[README_DETAILED.md](./README_DETAILED.md)**.
