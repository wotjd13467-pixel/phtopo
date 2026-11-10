# phtopo (command: `pt`)

Phonon topology from phonopy data: phonon dispersion, Weyl nodes with chirality χ,
nodal lines, Berry curvature, plus phonon Hall tools. One command `pt` with subcommands, like phonopy.

**If you are an AI agent: follow the "Recipe" section step by step. Do not skip the checks.**

---

## 0. Run environment (read first)

- Run on **Linux or WSL**. On this Windows machine the Application Control policy blocks
  `pt.exe` and the spglib DLL. If you see `애플리케이션 제어 정책에서 이 파일을 차단했습니다`
  (= "blocked by application control policy"), stop and use WSL instead.
- Check the install: `pt -h` must print a numbered list of commands. If `pt` is not found, use
  `python -m phtopo -h` (same thing).
- Needed Python packages: phonopy, spglib, seekpath, numpy, scipy, matplotlib, h5py, pyyaml.
  Optional: phono3py (`pt tau`, `pt 3rd`), ase + one of mace-torch / orb-models / sevenn (`pt relax`, `pt 2nd`, `pt 3rd`).
- Every command: `pt <command> -h` prints the full help. Read it if unsure.

## 1. Input files

Put these in ONE folder (the data folder). Names are exact:

| file | needed? | what it is |
|---|---|---|
| `POSCAR` | always | unit cell (VASP format) |
| `FORCE_SETS` | yes, unless you pass `--fc` | phonopy displacement forces |
| `BORN` | recommended | Born charges + dielectric tensor. If present, LO-TO (NAC) is ON automatically |

- The supercell is detected automatically from the number of atoms in `FORCE_SETS`
  (it prints e.g. `[슈퍼셀 3 3 3 — FORCE_SETS 324 원자]` = "supercell 3 3 3, 324 atoms").
  Only if it says it cannot guess (non-cubic supercell like 3×3×2), add `--d 3 3 2`.
- No `BORN` → it prints `[경고] BORN 없음 → NAC 끔` (= "no BORN, NAC off") and continues.
  For polar materials this changes the result; report it to the user.
- **Never write into the data folder.** Make a separate work folder, `cd` into it, and point to the
  data with `--dir <data folder>`. All outputs go to the current folder. Relative paths work: `--dir ../data`.

## 2. Recipe: find Weyl nodes and nodal lines

Replace `DATA` with the data folder path. Run each step, then do its check before the next step.

### Step 1 — dispersion sanity check (≈1 min)

```
pt band --dir DATA
```
Outputs `band.dat`, `band.png`. The k-path is chosen automatically.

Check the printed line `f <min> .. <max> thz`:
- If it prints `[경고] 허수 진동수 ...` (= "imaginary frequency", min below −0.1 THz):
  the structure is unstable or the force constants are bad. **Stop and report to the user.**
  Do not search for nodes on an unstable structure.
- Otherwise continue.

Remember the number of bands: 3 × (atoms in POSCAR). Band pairs are 1/2 … (3N−1)/3N.

### Step 2 — crossings on high-symmetry lines (≈2–3 min, all bands)

```
pt line --dir DATA
```
Outputs `line.txt` (table) and `line_nodes.txt` (point nodes only). Each row of `line.txt` ends with a verdict:

| verdict in `line.txt` | meaning |
|---|---|
| `Weyl χ = +1.000` / `-1.000` | Weyl point on the line |
| `nodal line 이 지나감 (점 아님)` | a nodal line crosses this path here (not a point) |
| `선 전체 축퇴 (강제)` | the whole segment is degenerate by symmetry (nodal surface / forced) |
| `회피 교차 (최소 갭 …)` | avoided crossing, no node |

### Step 3 — full Brillouin-zone search for Weyl points (main step)

First measure the time with one band pair from the middle of the spectrum
(M ≈ half of 3N, e.g. 17 for 36 bands; do not use 1–3, acoustic pairs are slow and atypical):
```
pt scan --dir DATA --band M --verify -o test.txt
```
Then run all band pairs (default when `--band` is not given):
```
pt scan --dir DATA --verify -o nodes.txt
```
- Full scan = (time of one band) × (3N−1). It can take hours. The output file is saved after EVERY
  band pair, so an interrupted run keeps finished bands. To limit the range use `--band 11-18`.
- **Choosing the q-mesh** (`--mesh`, `--mesh-max`). The search grid is mesh³ (symmetry-reduced).
  `--verify` doubles the mesh automatically for a band pair whose check fails, up to `--mesh-max`.

  | case | use |
  |---|---|
  | first run, primitive cell ≤ 20 atoms | defaults: `--mesh 24 --mesh-max 48` (nothing to add) |
  | primitive cell > 20 atoms (dense bands) | `--mesh 32 --mesh-max 64` |
  | a band pair still shows `불일치` after the run | rerun only that pair: `--band N --mesh 48 --mesh-max 96` |
  | just a quick look | `--mesh 16` without `--verify` |

  Cost grows like mesh³; doubling the mesh is ≈8× slower. These are rules of thumb, not guarantees.

**How to read the scan log (per band pair `b/b+1`):**

| log line | meaning | action |
|---|---|---|
| `b/b+1: 후보 X → 점 노드 Y (선/면 Z 버림, …)` | Y Weyl points found (symmetry-unique). Z candidates were lines/planes | Z > 0 → nodal lines or forced degeneracy exist for this pair (see step 5) |
| `강제 축퇴 k3=0.5` | whole plane k3=½ degenerate for this pair (symmetry) | not an error; it is a nodal surface |
| `(q1, q2, q3)  χ = -1.000 (r=…)  궤도 4` | one Weyl point, chirality, orbit size (number of symmetry copies) | — |
| `검사: … flux +1.000 … vs 찾은 노드 합 +1  OK` | check passed: no node missed in the check box | — |
| `<- 불일치` | check failed: a node was probably missed | handled automatically (bigger mesh). If still listed at the end, see below |
| `검사 불가` | check impossible (a nodal line or degeneracy touches the check box) | report "not verified" for this pair |
| `⚠ b/b+1: …` (at the very end) | unresolved warnings | report them; they are also written as `# 경고 …` lines in `nodes.txt` |

**`nodes.txt` format** (lines starting with `#` are comments):
```
#       q1          q2          q3     f(THz)     chi  band
    0.146493    0.070849    0.000000   10.0869  -1.000    17
```
- q1 q2 q3 = fractional coordinates of the reciprocal lattice of THIS POSCAR (rlu).
  The header line `# q = q1 b1 + q2 b2 + q3 b3 … b1=(…)` gives b1, b2, b3 in Cartesian 1/Å.
- `band` = lower band of the pair (node between band and band+1, sorted by frequency).
- Only one representative per symmetry orbit is written. The real number of nodes = sum of orbit sizes from the log.
- The Γ point (acoustic bands touching at zero frequency) is never reported as a node.

### Step 4 — confirm chirality when it is not an integer

If a `chi` value in `nodes.txt` is not close to ±1 (for example −0.4), nodes are probably clustered.
```
pt chi --dir DATA --nodes nodes.txt --radius 0.001
```
If `chirality.txt` prints `고립 아님 의심` (= "suspect not isolated"), use a box around the cluster:
```
pt box --dir DATA --nodes nodes.txt --node 1
```
(`--node 1` = first data row of nodes.txt.) The box result is the total χ of all nodes inside.

### Step 5 — nodal lines

`pt scan -o nodes.txt` also writes **`nodes_lines.txt`**: points where the gap is closed but the
degeneracy is NOT an isolated point (it lies on a nodal line or surface).
```
#       q1          q2          q3     f(THz)  band  plane
    0.000000    0.371912    0.001359    4.4054     6  k1=0
```
- Each row is a SAMPLE point on a line (several rows with the same `band` and `plane` = points on the same line).
  It is not the full curve. Only one symmetry representative is written.
- `plane` = the symmetry plane the point lies on (e.g. `k1=0` = mirror/glide plane, typical for nodal lines).
  `-` = general position.
- Report: for each band pair, the planes and the q ranges of its sample points.

Other signals:
- `line.txt` rows with `nodal line 이 지나감` → a nodal line crosses that high-symmetry segment.
- scan log `선/면 Z 버림` with Z > 0 → this band pair has line/plane degeneracies (Z = how many candidates fell on them).
- `강제 축퇴 kA=0.5` → symmetry-forced degenerate plane (nodal surface), not a line. Its points are not listed.
To see them, plot Berry curvature on a plane (needs `--band`):
```
pt berry --dir DATA --band 17 --axis 3
```

### Step 6 — report

Report: number of Weyl points per band pair (sum of orbit sizes), their q and χ, every `⚠` warning,
every `검사 불가` pair, nodal lines from step 5 (`nodes_lines.txt`: band, plane, q range), and whether NAC was on (BORN present).
The total χ over the whole Brillouin zone is always 0; do not report a nonzero total as a discovery.

## 3. No DFT forces? Use a machine-learned potential (MLIP)

Needs ase + mace-torch (or orb-models / sevenn). Without DFT there is **no BORN** → LO-TO is off.

```
pt relax --dir DATA --model mace                   # → POSCAR_relaxed (relaxed cell + ions)
pt 2nd -c POSCAR_relaxed --d 3 3 3 --model mace    # → fc2.hdf5 (second-order force constants)
pt band -c POSCAR_relaxed --fc fc2.hdf5 --d 3 3 3  # then the recipe with --fc instead of FORCE_SETS
pt 3rd -c POSCAR_relaxed --d 2 2 2 --model mace    # → fc3.hdf5 (third order, for lifetimes `pt tau`)
```
- `pt relax` options: `--fix-cell`, `--fix-symmetry`, `--fmax 0.01`, `--steps 300`. If it prints `미수렴` (= "not converged"), increase `--steps`.
- `pt 2nd` / `pt 3rd` restart from their checkpoint files if run again in the same folder, and skip if the output file already exists.
- There is **no fourth-order command** (`pt 4th` does not exist).
- With `--fc`, keep `--d` equal to the supercell used to make that file.

## 4. Errors and what to do

| message | do this |
|---|---|
| `슈퍼셀 원자 X 개인데 --d … — 아마 --d 3 3 3` | supercell mismatch → use the suggested `--d` |
| `POSCAR 없음` | wrong folder → fix `--dir` |
| `--dir: 폴더 '…' 없음` | the folder does not exist → fix the path |
| `--born: … 없음` / `--loto T: … 없음` | BORN file missing |
| `pt nodes: --band 필요` (also berry, simphony) | add `--band N` |
| `--band 36: 이 구조의 밴드쌍은 1..35` | band out of range |
| `--path: 이 구조에는 점 'M' 이 없음. 쓸 수 있는 이름: …` | use one of the listed point names |

## 5. Known problems and how to handle them

These happened in real runs. Match the symptom, then do exactly the "do this" column.

| symptom | likely cause | do this |
|---|---|---|
| several nodes within ~0.001 of each other, χ like −0.000 / +1 / −1 mixed, `r=0.0001` in the log | one degenerate point found several times, or nodal lines meeting | treat the cluster as ONE point. Run `pt box --node <row>` on it and report the box χ. Do not report each row as a separate Weyl point |
| a node with χ ≈ 0 (e.g. `-0.000`) on a line where two symmetry planes meet (two coordinates equal to 0 or 0.5) | symmetry-enforced crossing on a high-symmetry axis, not a Weyl point | report it as "χ=0 crossing on axis", not as Weyl |
| χ not close to an integer (e.g. −0.4, +0.7) | sphere too big (catches neighbours) or too small | step 4: `pt chi --radius 0.001`, then `pt box` |
| band pairs 1/2, 2/3, 3/4 (acoustic): slow, `불일치` (e.g. flux +2, found 0), or `검사 불가` | acoustic gaps are tiny; candidates slide into nodal lines | report "acoustic pair not verified". Do not rerun with ever-larger meshes |
| `불일치` remains after the run for a non-acoustic pair | node missed by the grid | rerun ONLY that pair once: `--band N --mesh 48 --mesh-max 96 -o nodes_N.txt`. If still `불일치`, report it as unresolved and stop |
| `검사 불가` | a nodal line or degenerate plane touches the check box | normal for some pairs; report "not verified" |
| `강제 축퇴 k3=0.5` | symmetry forces the whole plane k3=½ degenerate (screw axis) | not an error, not a node. Mention it as a nodal surface |
| many rows in `nodes_lines.txt` for one pair | it has nodal lines | expected for mirror/glide planes; report planes and q ranges |
| `[경고] 허수 진동수` in `pt band` | unstable structure, wrong supercell, or bad force constants | stop. Do not search for nodes. Report |
| `[경고] BORN 없음 → NAC 끔` | no BORN file | continue, but write in the report "LO-TO not included". Never create or guess a BORN file |
| the run was interrupted | — | the output file keeps the finished band pairs. Check its `band` column, then run the missing pairs with a NEW `-o` name (running with the same `-o` overwrites the file) |
| total χ of all nodes ≠ 0 | only symmetry representatives are listed; orbit copies are missing from the sum | not a discovery. The full-zone total is always 0 |
| MLIP route (`pt 2nd`) gives imaginary frequencies | MLIP error or structure not relaxed with the same model | run `pt relax` with the SAME `--model` first; if still imaginary, report that the MLIP is not reliable for this material |

**Do NOT:**
- edit any file in the data folder or any `.py` file of phtopo;
- invent options. Use only options shown by `pt <command> -h`;
- increase `--mesh` above 96 or rerun the same full scan again and again;
- run two scans that write the same `-o` file at the same time;
- report results from a run that printed an error or a traceback;
- convert the coordinates yourself; report q1 q2 q3 exactly as written (they are in this POSCAR's reciprocal-lattice units).

## 6. Other commands (short)

`pt hr` (tight-binding Hamiltonian, needs BORN) · `pt nodes --band N` (nodes on one plane only, fast) ·
`pt simphony --band N` (inputs for the Simphony code) · phonon Hall: `pt nlhall`, `pt spectral`, `pt tau`, `pt pprime`, `pt kappa`.
Common options for all: `--dir`, `--d`, `-c`, `--fs`, `--fc`, `--born`, `--loto T|F`, `--band`.
Full Korean manual and verification log: `../PHTOPO.md`.
