# HarnessBrain

Automated evolution framework that discovers effective memory systems and agent
scaffolds for two task families:



## Project Structure

```
HarnessBrain/
├── main.py                    # Entry point: evolve / benchmark (routes by --task)
├── run.sh                     # One-command: evolve then show results
├── config.yaml                # Datasets, models, concurrency, memory_config
├── .env                       # API keys (not committed)
├── .claude/skills/
│   ├── classification/        # Skill: one iteration of memory-system evolution
│   └── agent/                 # Skill: one iteration of agent-scaffold evolution
├── scripts/
│   ├── download_data.py       # Dataset downloader (text task)
│   └── run_eval.sh            # Harbor eval runner (agent task)
├── src/
│   ├── evolve.py              # Unified evolution loop (text + agent branches)
│   ├── benchmark.py           # Async job runner & results printer
│   ├── claude_wrapper.py      # Claude Code CLI wrapper (proposer)
│   ├── memory/                # Evolution memory: encode → update → steer
│   │   ├── encoding/          # encode runs/diffs into structured evidence
│   │   ├── updating/          # update evidence weights / utility
│   │   ├── steering/          # steer the next proposal from memory
│   │   └── reporting/         # offline metrics
│   ├── agents/                # Text-classification memory systems
│   └── harness_agents/        # Agent-task scaffolds: baselines + best evolved agents
├── data/                      # Downloaded datasets (gitignored)
├── logs/                      # Val results & evolution workspace
├── jobs/                      # Harbor job outputs (agent task)
└── results/                   # Test results
```


## Quick start

### 1. Install

```bash
# classification task
conda create -n harnessbrain python=3.11 -y
conda activate harnessbrain
pip install -r requirements.txt

# agent task (Terminal-Bench / SWE-bench) — requires Python 3.12 + Docker
conda create -n harnessbrain-agent python=3.12 -y
conda activate harnessbrain-agent
pip install -r requirements.txt -r requirements-terminal.txt
```

### 2. Configure

```bash
cp .env.example .env        # set CLASSIFIER_* (classification) and HARBOR_MODEL (agent)
claude                      # log in once — the proposer uses the Claude Code CLI
```

### 3. Run the classification task

```bash
python scripts/download_data.py   # one-time dataset download
./run.sh                                 # evolve, then print results
```

### 4. Run the agent task

```bash
# choose the dataset in config.yaml:  terminal_dataset: terminal-bench@2.0 | swebench-verified
docker info >/dev/null || colima start   # Docker must be running (Docker Desktop or colima)
./run.sh terminal                        # evolve; results in logs/frontier_val.json
```

No manual dataset download for the agent task: Harbor resolves the dataset
(`terminal-bench@2.0` / `swebench-verified`) and pulls each task's Docker image from
Docker Hub on first use. `run.sh terminal` launches Harbor for you (`harbor run`,
installed by `--extra terminal`) — no separate Harbor server to start.
