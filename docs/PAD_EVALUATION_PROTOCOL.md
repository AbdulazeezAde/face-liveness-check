# PAD evaluation and calibration protocol

This package can collect and calibrate **score-only** PAD observations. It does
not make a PAD model suitable for production on its own, and it must not be used
to create a biometric dataset without a lawful basis, explicit consent where
needed, and a documented retention policy.

## Collection protocol

1. Obtain documented consent and assign each participant an opaque sample ID.
   Do not encode names, document numbers, paths, or other identifiers in it.
2. Collect genuine, print, replay, and mask presentations on every camera,
   lighting condition, and client platform to be supported. Include the attack
   media and display technologies relevant to the deployment.
3. Keep source video and facial images out of the JSONL result file. The
   `evaluate-webcam` command stores only IDs, labels, timestamps, face counts,
   and model scores.
4. Split by person and capture session before calibration. A participant or
   device must never appear in both calibration and final evaluation data.
5. Independently review accessibility, false-reject, and demographic impacts.
   Define a manual-review / fallback route before changing any threshold.

## Calibration

The CLI averages repeated frames with the same opaque `sample_id`, so a long
recording does not outweigh another sample. It refuses to propose a threshold
until each of the genuine, print, replay, and mask groups has enough distinct
samples (20 by default).

```powershell
face-liveness-check calibrate-pad --input .\pad-scores.jsonl `
  --candidate facenox_experimental `
  --target-genuine-accept-rate 0.95 `
  --target-attack-reject-rate 0.95 `
  --minimum-samples-per-label 20
```

Save the command output, data collection protocol, model version/checksum,
camera and client versions, dataset split, and approval decision. Confirm the
proposed threshold on a separate held-out set before use. If the command returns
no eligible threshold, do not loosen the deployment policy merely to force one:
collect more representative data or route the outcome for review.

## Threshold approval record

Record, at minimum:

- Model pack/version/checksum and preprocessing configuration.
- Candidate threshold and sample-level genuine acceptance / attack rejection.
- Counts for genuine, print, replay, and mask groups; camera/client coverage.
- Collection dates, consent/lawful-basis reference, retention schedule, and
  deletion confirmation.
- Holdout evaluation result, accessibility review, reviewer, and expiry date.

Re-run calibration after changing camera hardware, client capture code, model,
crop/alignment pipeline, or intended attack classes.
