import os
import urllib.request
import zipfile
import gzip

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"📥 Downloading {url} -> {dest}...")
        urllib.request.urlretrieve(url, dest)
        print(f"✅ Downloaded {os.path.getsize(dest):,} bytes.")
    else:
        print(f"ℹ️ {dest} already exists ({os.path.getsize(dest):,} bytes).")


def prepare_datasets():
    # 1. High-density technical and legal corpus (already available on system)
    target_corpus = os.path.join(DATA_DIR, "corpus_technical.txt")
    if not os.path.exists(target_corpus):
        print("Creating technical text corpus from system sources...")
        import glob

        content = bytearray()
        # Collect rich diverse text from documentation and licenses
        for path in glob.glob("/usr/share/doc/**/*", recursive=True):
            if os.path.isfile(path) and not os.path.islink(path) and os.path.getsize(path) < 500000:
                try:
                    if path.endswith(".gz"):
                        with gzip.open(path, "rb") as f:
                            data = f.read()
                    else:
                        with open(path, "rb") as f:
                            data = f.read()
                    # Filter ascii/utf-8 text
                    if b"\x00" not in data[:2048] and len(data) > 1000:
                        content.extend(data[:50000])
                    if len(content) >= 200_000:  # 200 KB slice
                        break
                except Exception:
                    continue

        with open(target_corpus, "wb") as f:
            f.write(content)
        print(f"✅ Saved corpus_technical.txt: {len(content):,} bytes")

    # 2. Extract slice of enwik8 or download
    enwik_zip = os.path.join(DATA_DIR, "enwik8.zip")
    enwik_raw = os.path.join(DATA_DIR, "enwik8")
    slice_file = os.path.join(DATA_DIR, "enwik_slice.txt")

    if not os.path.exists(slice_file):
        try:
            download_file("https://mattmahoney.net/dc/enwik8.zip", enwik_zip)
            if os.path.exists(enwik_zip) and not os.path.exists(enwik_raw):
                with zipfile.ZipFile(enwik_zip, "r") as zip_ref:
                    zip_ref.extractall(DATA_DIR)

            if os.path.exists(enwik_raw):
                with open(enwik_raw, "rb") as f:
                    slice_data = f.read(100_000)  # 100 KB standard slice
                with open(slice_file, "wb") as f:
                    f.write(slice_data)
                print(f"✅ Prepared standard enwik_slice.txt: {len(slice_data):,} bytes")
        except Exception as e:
            print(f"⚠️ Could not download enwik8 over network: {e}. Using local diverse corpus.")


if __name__ == "__main__":
    prepare_datasets()
