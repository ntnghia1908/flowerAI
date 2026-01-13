# WSL Baseline Setup Guide

**Date:** 2026-01-11
**Status:** Setting up WSL environment for Flower baselines

---

## ✅ Solution: Use WSL for Linux Baselines

Since you have WSL (Windows Subsystem for Linux), we can run the official Flower baselines without modification!

**Benefits:**
- ✅ Official baseline implementations work as-is
- ✅ No platform compatibility issues
- ✅ Easy to cite in research papers
- ✅ Automatic updates from Flower team
- ✅ Keep Windows for development, Linux for experiments

---

## 📋 WSL Environment Setup

### Step 1: Verify WSL Installation

First, let's verify your WSL setup:

```bash
# From Windows PowerShell or CMD
wsl --version
wsl -l -v
```

**Expected output:**
- WSL version 2 (recommended)
- Ubuntu or other Linux distribution installed

### Step 2: Access WSL

```bash
# From Windows terminal
wsl
```

This will open a Linux shell.

### Step 3: Install Python 3.10 in WSL

```bash
# Update package list
sudo apt update
sudo apt upgrade -y

# Install Python 3.10 and pip
sudo apt install python3.10 python3.10-venv python3-pip -y

# Verify installation
python3.10 --version
pip3 --version
```

### Step 4: Install Poetry in WSL

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify Poetry installation
poetry --version
```

**Expected:** Poetry 1.8.0 or higher

---

## 📁 Data Access Strategy

### Option A: Access Windows Data from WSL (RECOMMENDED)

Windows drives are mounted in WSL at `/mnt/`:

```bash
# Your Windows data location
# Windows: C:\Users\DESKSTOP_003\Desktop\flowerAI\data
# WSL:     /mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/data

# Test access
ls /mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/data
```

**Advantages:**
- ✅ No data copying needed
- ✅ Single source of truth
- ✅ Save disk space
- ⚠️ Slightly slower I/O (usually not noticeable)

### Option B: Copy Data to WSL (Alternative)

If I/O performance is critical:

```bash
# Create workspace in WSL home
mkdir -p ~/flowerAI/data

# Copy data from Windows to WSL
cp -r /mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/data/* ~/flowerAI/data/

# Verify copy
ls ~/flowerAI/data
```

**Advantages:**
- ✅ Faster I/O performance
- ⚠️ Requires disk space (~2GB for all partitions)
- ⚠️ Need to sync changes

---

## 🔧 Baseline Setup in WSL

### Step 1: Clone Flower Baselines in WSL

```bash
# Navigate to workspace
cd /mnt/c/Users/DESKSTOP_003/Desktop

# Clone if not already cloned
# (You already cloned it on Windows, so it should be accessible)
ls flower_baselines/baselines

# OR clone fresh in WSL home
cd ~
git clone https://github.com/adap/flower.git flower_baselines
cd flower_baselines/baselines
```

### Step 2: Install Baseline Dependencies

For each baseline (FedNova, FedBN, FedPer, FedRep):

```bash
# Example: FedNova
cd ~/flower_baselines/baselines/fednova

# Install with Poetry
poetry install

# Verify installation
poetry run python -m fednova.main --help
```

Repeat for:
- `fedbn`
- `fedper`
- `fedrep`

---

## 📝 Baseline Modification Strategy

### Files to Modify for NPY Support

For each baseline, we need to:

1. **Add NPY data loader** (`task_npy.py`)
2. **Modify data loading** (support NPY mode in `dataset.py`)
3. **Add custom metrics** (precision, recall, f1 in client evaluate)
4. **Create metrics logger** (CSV output for your format)
5. **Create configs** (9 distributions + 1 test)

### Template Structure

```
baselines/[algorithm]/
├── [algorithm]/
│   ├── task_npy.py          # NEW: NPY data loader
│   ├── metrics_logger.py    # NEW: CSV output
│   ├── dataset.py            # MODIFY: Add NPY mode
│   ├── client.py             # MODIFY: Add custom metrics
│   └── strategy.py           # MODIFY: Add metrics aggregation
└── conf/
    ├── homo.yaml             # NEW: 9 distribution configs
    ├── C2.yaml
    ├── ... (7 more)
    └── test.yaml             # NEW: Test config
```

---

## 🎯 Implementation Plan (WSL)

### Phase 1: WSL Environment Setup (30-60 minutes)

- [x] Verify WSL installation
- [ ] Install Python 3.10 in WSL
- [ ] Install Poetry in WSL
- [ ] Verify data access from WSL
- [ ] Test baseline installation (FedNova)

### Phase 2: FedNova Migration (2-3 hours)

**Location:** `~/flower_baselines/baselines/fednova/`

**Files to create:**
1. `fednova/task_npy.py` - Copy from Windows project
2. `fednova/metrics_logger.py` - CSV output logger

**Files to modify:**
1. `fednova/dataset.py` - Add NPY support
2. `fednova/client.py` - Add precision, recall, f1 metrics
3. `fednova/strategy.py` - Add metrics aggregation callback

**Configs to create:**
1. `conf/homo.yaml`
2. `conf/C2.yaml`
3. `conf/C3.yaml`
4. `conf/C4.yaml`
5. `conf/C5.yaml`
6. `conf/Dir0.1.yaml`
7. `conf/Dir0.5.yaml`
8. `conf/Dir1.0.yaml`
9. `conf/Dir10.0.yaml`
10. `conf/test.yaml` (3 rounds)

**Test command:**
```bash
cd ~/flower_baselines/baselines/fednova
poetry run python -m fednova.main --config-name test
```

**Full run:**
```bash
# Run all 9 distributions using Hydra multirun
poetry run python -m fednova.main --multirun \
  distribution=homo,C2,C3,C4,C5,Dir0.1,Dir0.5,Dir1.0,Dir10.0
```

### Phase 3: FedBN Migration (2-3 hours)

**Location:** `~/flower_baselines/baselines/fedbn/`

Similar structure to FedNova, but:
- Uses Flower CLI (TOML configs instead of YAML)
- Run via: `flwr run . --run-config conf/[config].toml`

**Files to create:**
1. `fedbn/task_npy.py`
2. `fedbn/metrics_logger.py`

**Files to modify:**
1. `fedbn/dataset.py`
2. `fedbn/client_app.py`
3. `fedbn/strategy.py`

**Configs to create:** 9 TOML files + 1 test

**Test command:**
```bash
cd ~/flower_baselines/baselines/fedbn
flwr run . --run-config conf/test.toml
```

**Full run:**
```bash
# Run each distribution sequentially
for dist in homo C2 C3 C4 C5 Dir0.1 Dir0.5 Dir1.0 Dir10.0; do
  flwr run . --run-config conf/${dist}.toml
done
```

### Phase 4: FedPer Implementation (4-5 hours)

**Location:** `~/flower_baselines/baselines/fedper/`

**Additional features:**
- Personalized head layers (stay local)
- Base layers aggregate globally
- Client state persistence

**Files to create/modify:** Same as FedNova (Hydra-based)

### Phase 5: FedRep Implementation (4-5 hours)

**Location:** `~/flower_baselines/baselines/fedrep/`

**Additional features:**
- Two-phase training (representation then head)
- Representation layers aggregate

**Files to create/modify:** Same as FedBN (Flower CLI-based)

---

## 📊 Results Management

### Strategy 1: Save Results to Windows (RECOMMENDED)

Save results directly to Windows filesystem for easy access:

```python
# In metrics_logger.py
results_dir = "/mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/results/[algorithm]"
```

**Advantages:**
- ✅ Results immediately visible in Windows
- ✅ Easy to analyze with Windows tools
- ✅ Consistent with current project structure

### Strategy 2: Save to WSL, Copy Later

Save to WSL, then copy to Windows:

```bash
# Save results in WSL
results_dir = "~/flowerAI/results/[algorithm]"

# Copy to Windows after experiments
cp -r ~/flowerAI/results/* /mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/results/
```

---

## 🔄 Development Workflow

### Recommended Workflow:

1. **Development (Windows):**
   - Edit baseline files using VSCode on Windows
   - Files are at: `C:\Users\DESKSTOP_003\Desktop\flower_baselines\`

2. **Execution (WSL):**
   - Run experiments in WSL
   - Access files at: `/mnt/c/Users/DESKSTOP_003/Desktop/flower_baselines/`

3. **Results (Windows):**
   - Results saved to Windows filesystem
   - Analyze using Windows tools

**VSCode Remote-WSL Extension:**
- Install "Remote - WSL" extension
- Click "WSL" in bottom-left corner
- Open folder in WSL
- Edit and run in same environment!

---

## ✅ Verification Checklist

### WSL Environment:

- [ ] WSL version 2 running
- [ ] Python 3.10 installed
- [ ] Poetry installed
- [ ] Can access Windows data (`/mnt/c/Users/...`)

### Baseline Setup:

- [ ] FedNova baseline installed
- [ ] FedBN baseline installed
- [ ] FedPer baseline installed
- [ ] FedRep baseline installed

### Data Integration:

- [ ] `task_npy.py` copied to each baseline
- [ ] NPY data accessible from WSL
- [ ] Test data loading works

### Metrics Tracking:

- [ ] Precision, recall, f1 added to client evaluate
- [ ] Metrics aggregation in strategy
- [ ] CSV output format matches current project
- [ ] Hardware metrics captured

### Configuration:

- [ ] Test config (3 rounds) created
- [ ] 9 distribution configs created
- [ ] Data paths point to correct location

### Test Runs:

- [ ] FedNova test run (3 rounds) passes
- [ ] FedBN test run (3 rounds) passes
- [ ] FedPer test run (3 rounds) passes
- [ ] FedRep test run (3 rounds) passes

---

## 📋 Next Steps

### Immediate Actions:

1. **Verify WSL setup:**
   ```bash
   wsl --version
   wsl
   python3.10 --version
   ```

2. **Install dependencies in WSL:**
   ```bash
   # In WSL
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3-pip -y
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Test data access:**
   ```bash
   # In WSL
   ls /mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/data
   ```

4. **Install first baseline (FedNova):**
   ```bash
   # In WSL
   cd /mnt/c/Users/DESKSTOP_003/Desktop/flower_baselines/baselines/fednova
   poetry install
   ```

---

## 🎯 Timeline Estimate

**WSL Setup:** 30-60 minutes
**FedNova Migration:** 2-3 hours
**FedBN Migration:** 2-3 hours
**FedPer Implementation:** 4-5 hours
**FedRep Implementation:** 4-5 hours

**Total:** ~15 hours implementation + ~108 hours experiments

---

## 💡 Pro Tips

### 1. Use VSCode Remote-WSL

Edit files in Windows, run in WSL seamlessly:
- Install "Remote - WSL" extension
- Press `Ctrl+Shift+P` → "WSL: Connect to WSL"

### 2. Use tmux for Long Runs

Experiments take hours, use tmux to keep sessions alive:
```bash
# In WSL
sudo apt install tmux -y

# Start tmux session
tmux new -s experiments

# Run experiments
poetry run python -m fednova.main --multirun ...

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t experiments
```

### 3. Monitor from Windows

While experiments run in WSL, monitor from Windows:
```bash
# Windows PowerShell
wsl tail -f /mnt/c/Users/DESKSTOP_003/Desktop/flowerAI/results/FedNova/latest.log
```

### 4. Parallel Experiments

Run multiple experiments in parallel using tmux windows:
```bash
# Window 1: FedNova
tmux new -s fednova
poetry run python -m fednova.main --multirun ...

# Window 2: FedBN (Ctrl+B, C)
poetry run python -m fedbn.main --multirun ...
```

---

**Status:** ✅ WSL strategy documented, ready to begin setup
**Next:** Verify WSL environment and install Python dependencies
