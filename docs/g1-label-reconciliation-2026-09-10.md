# G1 — the two machines' label hashes, reconciled

**Measured 2026-09-10.** Gate G1 of
[`prereg-phase3-retrain-2026-09-10.md`](prereg-phase3-retrain-2026-09-10.md) §3. Nothing
in the label tree was written; every number below comes from a read-only pass.

---

## 1. Verdict

**G1 still fails, and it should. But it fails for a smaller and fully accounted reason
than the pre-registration states, and the two numbers the pre-registration compares are
not comparable to each other.**

* The local tree and `dgxanode01` differ by **exactly the night restore of 2026-09-02**
  — +94,553 boxes — and by **nothing else**. This is now measured, not assumed.
* The comparison written into G1 (`b92739202127` vs `287b11c50b5a`) puts a **tree**-scope
  hash against a **train**-scope hash. Those are different algorithms over different file
  sets. The correct train-vs-train comparison is `8ed69b5974ed` vs `287b11c50b5a`.
* The reconciliation action is therefore not an investigation. It is a transfer: the
  server holds pre-restore labels and must be brought to the restored tree before any
  Phase 3 training.

---

## 2. The scope error, and where it came from

`scripts/label_hash_ledger.py`'s own docstring warns that the three scopes are three
algorithms rather than three subsets, and `handoff-2026-09-04.md` §6 trap 1 records that
comparing across them "has happened twice." This is the third.

The origin is traceable. [`experiment-log-2026-09-02.md:711`](experiment-log-2026-09-02.md)
records the restore as:

> VIS **train** label hash `b92739202127…` (was `287b11c50b5a…`)

`b92739202127` is not the train hash. It is the **tree** hash — `restore_night_perbox.py`'s
byte-only directory walk over 127,309 files. The train hash after the restore is
`8ed69b5974ed` over 96,275 files, and the ledger has carried both correctly since
2026-09-04. The 2 September log mislabelled the scope, `prereg-night-label-restore.md:98`
inherited it, and G1 inherited it from there.

Nothing was decided on the strength of the wrong pair — both readings say "the machines
differ" — so no result is affected. But the pair as written would have sent whoever ran
G1 looking for a discrepancy of the wrong size.

### The three scopes, as measured today

| scope | hash | files | boxes | what it walks |
|---|---|---:|---:|---|
| `train` | `8ed69b5974ed` | 96,275 | 711,444 | the VIS train list; name + bytes + NUL |
| `all` | `df0cb307adc9` | 119,072 | 884,682 | train + val + test, same algorithm |
| `tree` | `b92739202127` | 127,309 | 962,960 | every `.txt` under the labels root; **bytes alone** |

`dgxanode01`'s recorded `287b11c50b5a` is a **`train`** hash — it is
`visfilter_manifest.json`'s `train_label_hash_after` and
`verify_dataset_state.py:65`'s `REF_LABEL_HASH_AFTER`.

---

## 3. What the difference actually is

The two trees cannot be diffed directly; the server's copy is not walkable from here
(Jupyter-only access, no SSH). So rather than compare hashes, the server's tree was
**reconstructed on this machine and hashed**.

The reconstruction is exact because `cut_dark` was blunt. It did not edit the affected
label files, it emptied them — `filter_night_boxes.py:358`, *"full cut: frame becomes
background"* — 17,502 of them, each named in `runs/visfilter/visfilter_manifest.json`.
The pre-restore tree is therefore today's tree with exactly those files blank.

`scripts/reconstruct_prerestore_hash.py` does this and reports:

```
manifest: 17,502 files emptied by cut_dark (132,688 boxes)
train images       : 96,275  (17,502 blanked)
reconstructed boxes: 616,891
reconstructed hash : 287b11c50b5a
server recorded    : 287b11c50b5a
VERDICT: MATCH -- the only difference is the night restore
```

**The reconstruction reproduces the server's hash bit-for-bit.** Since the hash folds in
every filename and every byte of every one of the 96,275 train labels, a match rules out
drift anywhere in the train tree, not merely in the night files.

The box arithmetic closes independently:

| state | train boxes | train hash |
|---|---:|---|
| pre-filter | 749,579 | `fd60c0834fdd` |
| after `--cut-dark` (**server today**) | 616,891 | `287b11c50b5a` |
| after `--restore` | 749,579 | — |
| after per-box re-drop of 38,135 (**local today**) | **711,444** | `8ed69b5974ed` |

711,444 − 616,891 = **94,553**, the restore's recorded net exactly.

---

## 4. What has to happen before G1 passes

G1 is a transfer, not a hunt:

1. Bring `dgxanode01` to the restored tree — either by copying the restored VIS labels or
   by re-running `filter_night_boxes.py --restore` followed by
   `restore_night_perbox.py --execute` against the committed
   `runs/visfilter/box_scores.csv`, which is what produced the local state.
2. Run `scripts/label_hash_ledger.py --scope both` **on the server** and record the row.
3. G1 passes when the server's `train` hash reads `8ed69b5974ed` and its `tree` hash reads
   `b92739202127`. Record both sides, per §3.

Until then §11 hard stop 1 holds: **no training starts.**

---

## 5. What this does not settle

* **It says nothing about the IR tree.** The reconstruction covers VIS train only, because
  that is the scope both recorded hashes use. IR labels were untouched by both operations
  (`experiment-log-2026-09-02.md:719`) but have never been hashed across the two machines
  at all. Stage 2 trains IR as `nc=1`, so an IR ledger scope is worth having before then;
  it does not exist today.
* **It is a statement about the tree, not about what any past run consumed.** A checkpoint
  trained on the server before 2026-09-02 saw the pre-restore labels whether or not the
  tree is reconciled now. That is what R-E1 slice 4's bracketing exists to bound, and it
  is not retroactive.
* **The 2026-09-03 21:19 rewrite is still unexplained.** 7,591 pohang01 train label files
  were rewritten with no script of this project running. The ledger bounds the window; it
  does not name a cause, and this reconciliation does not either.
