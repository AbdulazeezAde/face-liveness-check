# Evidence operations runbook

Evidence storage is opt-in and should remain exceptional. Configure it only for
sessions with the required consent and legal basis. The library does not send
evidence anywhere by default.

## Local encrypted retention

`LocalEncryptedEvidenceSink` encrypts metadata and NPY artifacts with Fernet.
Put the Fernet key in a secret manager separate from the evidence directory.
Each stored event may contain an explicit `retention_days` value. First run a
non-mutating preview; only then make deletion explicit:

```powershell
$env:FACE_LIVENESS_EVIDENCE_KEY = "<key supplied by your secret manager>"
face-liveness-check evidence-retention --evidence-local-dir .\encrypted-evidence
face-liveness-check evidence-retention --evidence-local-dir .\encrypted-evidence --apply
```

The default command is a dry run. The library deletes only directories which
have a valid encrypted event record, whose opaque ID matches the directory, and
whose recorded deadline has passed. Missing, malformed, or indefinite-retention
records are skipped and must be investigated. Capture the JSON output in an
access-controlled audit log, including the operator, command version, and ticket
reference.

## Fernet key rotation

Key rotation is an operational migration, not a setting change. Before rotation:

1. Inventory the encrypted evidence and verify backups under access controls.
2. Generate a new key in the approved secret manager and grant access only to
   the rotation job.
3. Test decryption and re-encryption on a separately consented, non-production
   copy. Never test by copying production evidence to an unmanaged machine.
4. Re-encrypt atomically with a purpose-built job that retains the old key until
   every object is verified with the new key. Record old/new key identifiers,
   object counts, checksums, time, and operator; never log key material.
5. Remove old-key access only after audit approval and the rollback window.

The package intentionally does not automate a multi-file Fernet key rotation:
an interrupted bulk rewrite could leave a directory encrypted with mixed keys.

## S3 lifecycle and IAM

`S3EvidenceSink` writes every object with SSE-KMS. Apply a lifecycle rule to the
specific evidence prefix; choose the retention duration to match the product
policy. Example (replace placeholders):

```json
{
  "Rules": [{
    "ID": "expire-face-liveness-evidence",
    "Status": "Enabled",
    "Filter": {"Prefix": "face-liveness-evidence/"},
    "Expiration": {"Days": 30},
    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}
  }]
}
```

Grant the application only `s3:PutObject` for that prefix and KMS encrypt/data
key permissions for the chosen key. Grant deletion and lifecycle administration
to a separate, audited operations role. Do not grant broad bucket listing,
public access, ACL changes, or decrypt permissions to the application role.

Audit at least quarterly: lifecycle state, KMS key policy, IAM access, object
prefix, retention output, consent records, and successful deletion evidence.
