# ID document extraction

`IdDocumentExtractor` is an optional local-processing layer for document
normalisation, OCR, template extraction, and portrait crops. It is not a proof
that a document is authentic, unaltered, or belongs to the person presenting it.

Install its optional runtime dependencies:

```bash
pip install "face-liveness-check[id-ocr]"
```

## First supported template: passport TD3 MRZ

The initial template extracts the two 44-character passport MRZ lines and
validates ICAO check digits for document number, birth date, expiry date,
personal number, and composite value. This makes it a safer first format than
guessing field locations on every national ID design.

```python
from face_liveness_check import IdDocumentExtractor, PaddleOcrEngine

extractor = IdDocumentExtractor(PaddleOcrEngine())
result = extractor.extract("passport.pdf")

if result.requires_manual_review:
    route_to_review(result.warnings)
else:
    portrait = result.portrait_crop_bgr
    document_number = result.fields["document_number"].value
```

The extractor returns all text, field values, OCR blocks, and portrait pixels in
memory. It does not write the input file, rectified document, or extracted PII.
`return_normalized_document=True` is intentionally explicit because it returns
a complete, sensitive document image to the caller.

## Configurable card templates and barcodes

For a versioned card layout, supply labels, required fields, and validators.
This is a building block for country-specific templates; it is not a claim that
all current or future national cards share the same layout.

```python
from face_liveness_check import LabelledCardTemplate

template = LabelledCardTemplate(
    {"document_number": ("DOCUMENT NO", "ID NO"), "full_name": ("FULL NAME",)},
    required_fields=("document_number", "full_name"),
    validators={"document_number": lambda value: value.isalnum() and 6 <= len(value) <= 24},
)
```

`ZxingBarcodeReader` optionally decodes local QR and PDF417 payloads. The
default CLI reports only barcode formats/counts; it requires the explicit
`--include-barcode-text` flag to print sensitive payload contents. Use a
document-specific parser to compare barcode claims with OCR/MRZ fields and send
disagreements to manual review.

```bash
face-liveness-check extract-id passport.pdf --document-type passport_td3 --read-barcodes
```

## OCR runtime and model provenance

The default adapter uses the PaddleOCR 3.x `PaddleOCR(...).predict(...)` API
with its document preprocessor disabled because this package performs its own
OpenCV perspective correction. PaddleOCR supports local inference engines and
document-orientation/unwarping options; model selection and local model
directories can be controlled by passing a configured PaddleOCR runner to
`PaddleOcrEngine`. See the [official PaddleOCR Python quick start](https://www.paddleocr.ai/main/en/quick_start.html).

PaddleOCR's own automatic model download is convenient for development, but it
does **not** meet this package's checksum-pinned model-pack policy. Production
integrators should provision approved model directories from a reviewed internal
artifact repository, configure PaddleOCR to use those directories, and record
model versions/checksums in their deployment configuration. A first-party
`id-ocr-default` model pack will only be added after its upstream model files,
licences, immutable URLs, and SHA-256 checksums have been reviewed.

## Adding card templates

Add one template per document version—not a generic claim that every card from a
country has the same layout. A template should define:

- Expected document markers and layout regions.
- Field extraction/normalisation rules and confidence policy.
- Date and document-number validation patterns.
- Portrait region or face-detection constraints.
- Barcode/MRZ comparison rules where an independent source exists.
- Manual-review triggers for glare, blur, missing fields, conflicts, and low
  confidence.

Do not commit real IDs, OCR output containing real identities, or live portrait
crops as test fixtures. Use generated records and consented test material under
an explicit retention policy.
