"""Download the original public Kaggle archive without inspecting credentials."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
EXPECTED = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def main():
    target = Path("data/raw")
    target.mkdir(parents=True, exist_ok=True)
    csv = target / "creditcard.csv"
    if not csv.exists():
        archive = target / "creditcard.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            # Extract only the expected member, never arbitrary archive paths.
            with zipped.open("creditcard.csv") as source, csv.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
    digest = hashlib.file_digest(csv.open("rb"), "sha256").hexdigest()
    if digest != EXPECTED:
        raise ValueError(f"Source checksum mismatch: {digest}")
    Path("data/provenance.json").write_text(
        json.dumps(
            {
                "source": URL,
                "sha256": digest,
                "dataset": "ULB/Worldline Credit Card Fraud Detection",
                "source_page": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                "note": "No raw data redistribution; verifies the observed source version.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {csv}: {digest}")


if __name__ == "__main__":
    main()
