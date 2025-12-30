# Batch Input Formats

The `/api/encode/batch` endpoint supports multiple input formats for encoding multiple sequences at once.

---

## JSON Format

Array of objects with `id` and `sequence` fields.

**Request:**
```json
{
  "format": "json",
  "sequences": [
    {"id": "gene1", "sequence": "ATGGCTCTGTGG"},
    {"id": "gene2", "sequence": "ATGAAAGAGTGA"},
    {"id": "gene3", "sequence": "ATGCCCAAATTT"}
  ]
}
```

**Notes:**
- `id` is optional (will be auto-generated as `seq_1`, `seq_2`, etc.)
- Sequences are cleaned (non-ATCG characters removed)

---

## FASTA Format

Standard FASTA format with headers starting with `>`.

**Request:**
```json
{
  "format": "fasta",
  "sequences": ">gene1 Human insulin\nATGGCTCTGTGGATGCGCCTG\n>gene2 Beta-actin\nATGGATGATATCATCATCATC\n>gene3\nATGCCCAAATTT"
}
```

**Parsed as:**
| ID | Description | Sequence |
|----|-------------|----------|
| gene1 | Human insulin | ATGGCTCTGTGGATGCGCCTG |
| gene2 | Beta-actin | ATGGATGATATCATCATCATC |
| gene3 | | ATGCCCAAATTT |

**Notes:**
- Header line starts with `>`
- ID is first word after `>`
- Description is rest of header (optional)
- Sequence can span multiple lines
- Whitespace in sequences is ignored

---

## CSV Format

Comma-separated values with header row.

**Request:**
```json
{
  "format": "csv",
  "sequences": "id,sequence,description\ngene1,ATGGCTCTGTGG,Human insulin\ngene2,ATGAAAGAGTGA,Beta-actin\ngene3,ATGCCCAAATTT,"
}
```

**Required columns:**
- `sequence` - DNA sequence (required)

**Optional columns:**
- `id` - Sequence identifier
- `description` - Description text
- `name` - Alternative to `id`

**Notes:**
- First row must be headers
- Column order doesn't matter
- Extra columns are ignored
- Handles quoted fields with commas

---

## TSV Format

Tab-separated values (same as CSV but with tabs).

**Request:**
```json
{
  "format": "tsv",
  "sequences": "id\tsequence\ngene1\tATGGCTCTGTGG\ngene2\tATGAAAGAGTGA"
}
```

---

## File Upload

For large batches, you can upload files directly.

**cURL Example:**
```bash
# FASTA file
curl -X POST "http://localhost:8765/api/encode/batch" \
  -F "file=@sequences.fasta" \
  -F "format=fasta"

# CSV file
curl -X POST "http://localhost:8765/api/encode/batch" \
  -F "file=@sequences.csv" \
  -F "format=csv"
```

---

## Response Format

All batch requests return the same response structure:

```json
{
  "results": [
    {
      "id": "gene1",
      "description": "Human insulin",
      "sequence": "ATGGCTCTGTGG",
      "protein": "MAL",
      "encoded": [
        {
          "seq_position": 0,
          "codon": "ATG",
          "amino_acid": "M",
          "depth": 1,
          "confidence": 0.89,
          ...
        },
        ...
      ],
      "stats": {
        "length": 4,
        "mean_depth": 1.25,
        "depth_std": 0.5,
        "mean_confidence": 0.87,
        "mean_margin": 0.42
      }
    },
    ...
  ],
  "summary": {
    "total_sequences": 3,
    "total_codons": 12,
    "failed_sequences": 0,
    "processing_time_ms": 45
  },
  "errors": []
}
```

---

## Limits

| Parameter | Limit |
|-----------|-------|
| Max sequences per batch | 1000 |
| Max sequence length | 100,000 bp |
| Max total request size | 10 MB |
| Timeout | 60 seconds |

---

## Examples

### Python
```python
import requests

# JSON format
response = requests.post(
    "http://localhost:8765/api/encode/batch",
    json={
        "format": "json",
        "sequences": [
            {"id": "seq1", "sequence": "ATGGCTCTGTGG"},
            {"id": "seq2", "sequence": "ATGAAAGGG"}
        ]
    }
)
results = response.json()
```

### JavaScript
```javascript
const response = await fetch("http://localhost:8765/api/encode/batch", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    format: "fasta",
    sequences: ">seq1\nATGGCTCTGTGG\n>seq2\nATGAAAGGG"
  })
});
const results = await response.json();
```

### cURL
```bash
curl -X POST "http://localhost:8765/api/encode/batch" \
  -H "Content-Type: application/json" \
  -d '{"format":"json","sequences":[{"id":"seq1","sequence":"ATGGCTCTGTGG"}]}'
```
