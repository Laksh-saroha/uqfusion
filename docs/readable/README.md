# `docs/readable/` — non-canonical prose editions of the pre-registrations

**Nothing in this directory is a pre-registration. Do not cite a file here.**

## What these are

On 2026-09-08/09 an editorial pass ran over the whole of `docs/`: roughly 15% shorter,
tighter prose, no intended change of meaning ("which fires the audit's first branch" →
"firing the audit's first branch"; "roughly seven times" → "~7×"). For handoffs,
experiment logs and TODOs that pass was kept and is committed in place.

It also touched all eight pre-registrations. Those were reverted to their committed text,
and the edited versions were moved here so nothing was lost.

## Why the originals were restored

A pre-registration's value is that its text is the text that existed **before** the run.
Once it has been rewritten afterwards, a reader cannot tell rewording from
re-specification without diffing against git — which is precisely the work the document
exists to make unnecessary.

The pass was also not neutral on this point. It deleted the sentence *"Committed ahead of
the run so the rule is verifiable in git history"* from `prereg-night-veto.md` and
`prereg-night-label-restore.md`, truncated the same claim in
`prereg-snms-draw-averaged-gate.md`, and in `prereg-uq-day-night-slice.md` turned
*"**Fixed** before the run, not to be edited afterwards"* into a parenthetical. Four
preregs lost their immutability assertion, to an edit that breaks it. That is a pattern of
the pass rather than a judgement about anyone's intent, and it is the reason the eight were
handled differently from the other 41 documents.

## How to use them

* Read a `*-readable.md` if the canonical text is heavy going.
* **Verify every rule, threshold, band and consequence against the canonical
  `docs/prereg-*.md`.** Where the two differ, the canonical file wins, without argument.
* These files are frozen. They are not updated when a prereg is amended — an amendment is
  its own committed document, per the standing rule that a rule change is declared and
  never edited in place.
