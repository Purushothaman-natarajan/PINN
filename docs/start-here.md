# Start here (no experience needed)

Never used a terminal? Perfect — this page assumes **zero background** and
takes you step by step from an empty computer to your first simulation
result. It takes about 20 minutes, most of it waiting for downloads.

> **How to use this page:** do the steps **in order**. After each command,
> compare your screen with the *“You should see”* box. If it doesn't match,
> read the *“If it fails”* note right below — don't skip ahead.

## Step 0. What is a terminal?

A terminal is a window where you type text commands and the computer runs
them. (Programmers also call it a *console*, *shell*, or *command line* —
same thing.) You will type lines like `python --version`, press ++enter++,
and read what the computer prints back. That's all there is to it.

**Open one now:**

=== "Windows"

    Press ++win+r++, type `powershell`, press ++enter++. A blue window opens.
    (Windows Terminal from the Microsoft Store also works and looks nicer.)

=== "macOS"

    Press ++cmd+space++, type `Terminal`, press ++enter++. A white window opens.

=== "Linux"

    Press ++ctrl+alt+t++. A window opens.

You should see a prompt ending in `>` or `$` or `%` — that means the terminal
is waiting for you. Words in `this style` in the steps below are things you
type. Never type the prompt character itself.

## Step 1. Install Python (the language this project speaks)

The project needs **Python 3.10 or newer**. Check if you already have it —
type this and press ++enter++:

```bash
python --version
```

*You should see:* something like `Python 3.12.4`. Any number **3.10 or
higher** is fine — move to Step 2.

*If it fails* (`not recognized`, `command not found`, or a lower number):

=== "Windows"

    1. Go to [python.org/downloads](https://www.python.org/downloads/), download
       the latest Python 3 release, and run the installer.
    2. ⚠️ On the **very first installer screen**, tick the box
       **“Add python.exe to PATH”** before clicking Install. (Forgetting
       this is the #1 cause of `python is not recognized`.)
    3. Close your terminal, open a fresh one (Step 0), and try
       `python --version` again.

=== "macOS"

    1. Go to [python.org/downloads](https://www.python.org/downloads/),
       download the latest Python 3 release, and run the `.pkg` installer.
    2. If `python --version` still fails but `python3 --version` works, use
       `python3` (and `pip3`) everywhere below instead of `python` (and `pip`).

=== "Linux"

    ```bash
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip
    ```
    Then use `python3` instead of `python` in every step below.

Unfamiliar words? Every one is explained in plain language in the
[Glossary](glossary.md).

## Step 2. Install Git (the download tool for code)

Git downloads the project from GitHub. Check for it:

```bash
git --version
```

*You should see:* something like `git version 2.45.0`. Any version is fine.

*If it fails:* install it, then **close and reopen the terminal**:

- Windows: [git-scm.com/download/win](https://git-scm.com/download/win) → run the installer, accept all defaults.
- macOS: [git-scm.com/download/mac](https://git-scm.com/download/mac).
- Linux: `sudo apt install -y git`.

## Step 3. Download the project

Type these two lines, one at a time (the first downloads ~a few MB, the
second steps *into* the project folder):

```bash
git clone https://github.com/Purushothaman-natarajan/PINN.git
cd PINN
```

*You should see:* `Cloning into 'PINN'...` followed by `done.`, and your
prompt now ends with `PINN`. Everything below happens inside this folder —
if you close the terminal later, come back with `cd PINN` (navigating to
wherever you ran the clone).

## Step 4. Create a private workspace (virtual environment)

This gives the project its own isolated set of libraries so nothing on your
computer gets disturbed:

=== "Windows"

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

*You should see:* your prompt gains a `(.venv)` prefix, e.g.
`(.venv) C:\Users\you\PINN>`. That prefix means "the workspace is active".
You need it active every time you work — if it's missing, just run the
`activate` line again (no need to redo anything else).

## Step 5. Install the project's libraries

```bash
pip install -r requirements.txt
```

*You should see:* lots of `Collecting ...` / `Installing ...` lines for a
few minutes, ending without red `ERROR` text.

*If it fails* with `error: externally-managed-environment` (some Linux
systems): you skipped Step 4 — the `(.venv)` prefix must be showing.
Activate the workspace and retry.

## Step 6. Run the self-test (proves everything works)

```bash
pytest -q
```

*You should see:* a line ending in something like `25 passed`. 🎉 Your
setup is correct.

*If tests fail:* copy the last 20 lines of the red output and look them up
in [FAQ & troubleshooting](faq.md) — or open a
[GitHub issue](https://github.com/Purushothaman-natarajan/PINN/issues)
and paste them there.

## Step 7. Run your first simulation (30 seconds)

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
```

What just happened: the computer solved the fluid-flow equations with a
traditional numerical method and saved the answer to
`data/raw/baseline.npz`. *You should see:* `Baseline saved to ...`.

**Look at your result.** The answer is four number-lists (velocity,
two temperatures, concentration) along 400 points. Peek at it:

```bash
python -c "import numpy as np; z = np.load('data/raw/baseline.npz'); print(list(z.files)); print('points:', len(z['eta'])); print('theta_f goes', round(float(z['theta_f'][0]), 2), '->', round(float(z['theta_f'][-1]), 2))"
```

*You should see:* the file list, `points: 400`, and `theta_f goes 1.0 ->
0.0` — temperature falling from the hot wall to the outside, exactly as
physics demands.

## Step 8. Train your first neural network (a few minutes)

This is the real thing: a neural network learning the same physics.
To keep it short we use a small practice setup:

=== "Windows"

    ```bash
    copy configs\default_trihybrid_ltne.yaml configs\my_first.yaml
    ```

=== "macOS / Linux"

    ```bash
    cp configs/default_trihybrid_ltne.yaml configs/my_first.yaml
    ```

Open `configs/my_first.yaml` in any text editor (Notepad works), find the
`training:` section, and change two numbers:

```yaml
training:
  epochs: 200        # was 20000
  log_every: 50      # was 500
```

Save, then run:

```bash
python main.py --config configs/my_first.yaml --mode train
```

*You should see:* lines like `[epoch 000050/...] total=...` with the
numbers shrinking — the network is learning. When it finishes:

```bash
python main.py --config configs/my_first.yaml --mode validate
```

*You should see:* a table of MSE/RMSE/R² numbers per field plus
`Accuracy gate R2>0.95: FAIL`. **Don't worry — FAIL is expected here!**
200 epochs is a practice run; the full 20 000-epoch training passes the
gate (our reference run reached R² = 0.9997). The pictures land in
`data/processed/figures/` — open `profiles.png` to see your network's
curves against the numerical answer.

!!! tip "Delete the practice file when done"
    `configs/my_first.yaml` is yours — it won't be committed (only tracked
    files go to git). The full-quality run is simply
    `python main.py --config configs/default_trihybrid_ltne.yaml --mode all`
    (takes hours on CPU — start it before lunch).

## Where next?

| I want to… | Go to |
|---|---|
| Understand what each command did | [Quickstart](quickstart.md) (same flow, with explanations) |
| Learn the science + papers behind it | [Learn the methods](learn.md) |
| See what data files exist | [Datasets & mock data](datasets.md) |
| Change parameters or equations | [Configuration reference](configuration.md) |
| Fix something broken | [FAQ & troubleshooting](faq.md) |
| Look up a strange word | [Glossary](glossary.md) |
