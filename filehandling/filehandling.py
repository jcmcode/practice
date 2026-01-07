"""
================================================================================
FILE HANDLING IN PYTHON - COMPREHENSIVE CHEAT SHEET
================================================================================
A practical guide to reading, writing, and manipulating files in Python.
Covers pathlib, text/binary files, JSON/CSV, compression, and best practices.

CONTENTS:
1. Pathlib Essentials: Modern cross-platform path handling
2. Basic Text File Operations: read, write, append
3. Working with Binary Files: bytes, images, serialized data
4. JSON & YAML: Structured data serialization
5. CSV & TSV: Tabular data handling
6. XML & HTML: Markup files
7. Directory Operations: listing, walking, globbing
8. File Metadata: size, modification time, permissions
9. Temporary Files: safe temp file creation
10. Large File Handling: efficient streaming & memory management
11. Compression: zip, gzip, tar archives
12. File Locking & Concurrency: safe multi-access patterns
13. Common Utilities: file comparison, finding, monitoring
================================================================================
"""

import os
import sys
import json
import csv
import shutil
import hashlib
from pathlib import Path
from io import StringIO, BytesIO
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Any
import tempfile


# ============================================================================
# 1. PATHLIB ESSENTIALS
# ============================================================================

def pathlib_basics():
    """Modern, object-oriented path handling (prefer over os.path)"""
    
    # Create Path objects
    p = Path("data/file.txt")           # Relative path
    home = Path.home()                   # User home directory
    cwd = Path.cwd()                     # Current working directory
    root = Path("/")                     # Absolute root
    
    # Path construction
    data_dir = Path("data")
    file_path = data_dir / "subdir" / "file.txt"  # Use / operator
    
    # Useful properties
    name = file_path.name                # "file.txt" (filename only)
    stem = file_path.stem                # "file" (without extension)
    suffix = file_path.suffix            # ".txt" (extension)
    parent = file_path.parent            # "data/subdir" (parent directory)
    
    # Check existence and type
    is_file = file_path.is_file()
    is_dir = file_path.is_dir()
    exists = file_path.exists()
    
    # Create directories
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Make parent dirs too
    
    # Absolute vs resolved path
    absolute = file_path.absolute()      # May contain .. and symlinks
    resolved = file_path.resolve()       # Canonical absolute path
    
    # Relative path computation
    try:
        rel = file_path.relative_to(Path.cwd())
    except ValueError:
        rel = None  # path is not relative to cwd
    
    print(f"Path: {file_path}")
    print(f"Name: {name}, Stem: {stem}, Suffix: {suffix}")


def pathlib_glob_and_search():
    """Find files using glob patterns"""
    base = Path(__file__).parent
    
    # Glob: find all matching files
    txt_files = list(base.glob("*.txt"))           # All .txt files in current dir
    py_files = list(base.glob("**/*.py"))          # All .py files recursively
    
    # Rglob: recursive glob (shorthand for **/ prefix)
    all_files = list(base.rglob("*"))              # All files recursively
    
    # Glob with patterns
    config_files = list(base.glob("config.*"))     # config.json, config.yaml, etc.
    logs = list(base.glob("logs/*.log"))           # Only direct children
    
    # Iterdir: list directory contents (non-recursive)
    contents = list(base.iterdir())
    files_only = [p for p in base.iterdir() if p.is_file()]
    dirs_only = [p for p in base.iterdir() if p.is_dir()]
    
    print(f"Found {len(txt_files)} txt files")


# ============================================================================
# 2. BASIC TEXT FILE OPERATIONS
# ============================================================================

def text_file_read_write():
    """Read and write text files with proper encoding handling"""
    base = Path(__file__).parent
    test_file = base / "test.txt"
    
    # WRITE: Simple write (overwrites file)
    test_file.write_text("Hello World\nLine 2\n", encoding="utf-8")
    
    # READ: Read entire file at once
    content = test_file.read_text(encoding="utf-8")
    print(f"Content: {content}")
    
    # READ: Line by line (memory efficient for large files)
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:  # Already stripped of \n by iteration
            line = line.rstrip("\n")
            print(f"Line: {line}")
    
    # APPEND: Add to end of file
    with open(test_file, "a", encoding="utf-8") as f:
        f.write("Appended line\n")
    
    # WRITE with explicit handle
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("New content\n")
        f.write("More content\n")


def text_file_advanced():
    """Advanced text file operations: seek, tell, modes"""
    base = Path(__file__).parent
    test_file = base / "test.txt"
    test_file.write_text("ABCDEFGHIJ\n0123456789\n", encoding="utf-8")
    
    # r+: read and write (file must exist)
    with open(test_file, "r+", encoding="utf-8") as f:
        content = f.read()           # Read all
        f.seek(0)                    # Go to start
        f.write("XY")                # Write at position 0 (overwrites AB)
    
    # w+: write and read (truncates file)
    with open(test_file, "w+", encoding="utf-8") as f:
        f.write("test data")
        f.seek(0)                    # Go back to start to read
        data = f.read()
        print(data)
    
    # Reading in chunks
    with open(test_file, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024)     # Read 1024 characters
            if not chunk:
                break
            print(f"Chunk: {chunk}")
    
    # Using readlines()
    lines = test_file.read_text(encoding="utf-8").splitlines()
    print(lines)


# ============================================================================
# 3. BINARY FILE OPERATIONS
# ============================================================================

def binary_file_operations():
    """Read/write binary data: images, executables, serialized objects"""
    base = Path(__file__).parent
    bin_file = base / "data.bin"
    
    # WRITE binary
    binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd"
    bin_file.write_bytes(binary_data)
    
    # READ binary
    raw = bin_file.read_bytes()
    print(f"Binary: {raw.hex()}")  # Print as hex
    
    # Read binary in chunks (for large files like images)
    with open(bin_file, "rb") as f:
        while True:
            chunk = f.read(1024)
            if not chunk:
                break
            # Process chunk (e.g., compute hash, transform data)
            print(f"Chunk size: {len(chunk)}")
    
    # Write binary incrementally
    with open(bin_file, "wb") as f:
        f.write(b"Start")
        f.write(b" ")
        f.write(b"of")
        f.write(b" ")
        f.write(b"file")


def file_hashing():
    """Compute checksums for file integrity verification"""
    base = Path(__file__).parent
    test_file = base / "test.txt"
    test_file.write_text("content", encoding="utf-8")
    
    # MD5 (not cryptographically secure, use for non-security purposes)
    def compute_hash(filepath, algorithm="sha256"):
        hasher = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):  # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    
    sha256 = compute_hash(test_file, "sha256")
    md5 = compute_hash(test_file, "md5")
    
    print(f"SHA256: {sha256}")
    print(f"MD5: {md5}")


# ============================================================================
# 4. JSON & YAML
# ============================================================================

def json_operations():
    """JSON serialization: dict/list to JSON and back"""
    base = Path(__file__).parent
    json_file = base / "data.json"
    
    # Object to JSON (dump)
    data = {
        "name": "Ada",
        "age": 36,
        "skills": ["Python", "Rust", "Go"],
        "meta": {
            "created": "2024-01-01",
            "verified": True
        }
    }
    
    # Write JSON to file
    json_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # Or use json.dump() with file handle
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Read JSON from file
    loaded = json.loads(json_file.read_text(encoding="utf-8"))
    
    # Or use json.load() with file handle
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    print(f"Loaded: {loaded}")
    
    # JSON Lines format (one JSON object per line, streamable)
    jsonl_file = base / "data.jsonl"
    records = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"}
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    # Read JSONL
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            print(f"Record: {obj}")


def yaml_operations():
    """YAML handling (requires PyYAML)"""
    try:
        import yaml
        base = Path(__file__).parent
        yaml_file = base / "config.yaml"
        
        # Python dict
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "mydb"
            },
            "debug": True,
            "workers": 4
        }
        
        # Write YAML
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Read YAML
        with open(yaml_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        
        print(f"YAML loaded: {loaded}")
    except ImportError:
        print("PyYAML not installed; pip install PyYAML")


# ============================================================================
# 5. CSV & TSV
# ============================================================================

def csv_operations():
    """CSV read/write with proper quoting and delimiter handling"""
    base = Path(__file__).parent
    csv_file = base / "data.csv"
    
    # WRITE CSV
    rows = [
        ["id", "name", "email", "active"],
        [1, "Alice", "alice@example.com", "True"],
        [2, "Bob", "bob@example.com", "False"],
        [3, 'Charlie "Doc" Smith', "charlie@example.com", "True"],  # Quoted name
    ]
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)
    
    # READ CSV
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # First row
        for row in reader:
            print(f"Row: {row}")
    
    # DictWriter: write with column names
    dict_rows = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "name", "email"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)
    
    # DictReader: read as dicts
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"Dict row: {row}")
    
    # TSV (Tab-Separated Values)
    tsv_file = base / "data.tsv"
    with open(tsv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def csv_advanced():
    """CSV with custom dialects and special handling"""
    base = Path(__file__).parent
    csv_file = base / "data.csv"
    
    # Custom dialect: strict quoting for safety
    csv.register_dialect("safe", quoting=csv.QUOTE_ALL, skipinitialspace=True)
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, dialect="safe")
        writer.writerow(["id", "name"])
        writer.writerow([1, "Alice"])
    
    # Pandas alternative (if pandas available)
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(df)
    except ImportError:
        print("pandas not installed")


# ============================================================================
# 6. XML & HTML
# ============================================================================

def xml_operations():
    """XML parsing and generation"""
    import xml.etree.ElementTree as ET
    
    base = Path(__file__).parent
    xml_file = base / "data.xml"
    
    # CREATE XML
    root = ET.Element("people")
    
    person1 = ET.SubElement(root, "person")
    person1.set("id", "1")
    name1 = ET.SubElement(person1, "name")
    name1.text = "Alice"
    email1 = ET.SubElement(person1, "email")
    email1.text = "alice@example.com"
    
    person2 = ET.SubElement(root, "person")
    person2.set("id", "2")
    name2 = ET.SubElement(person2, "name")
    name2.text = "Bob"
    email2 = ET.SubElement(person2, "email")
    email2.text = "bob@example.com"
    
    # Write to file
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    
    # PARSE XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    for person in root.findall("person"):
        person_id = person.get("id")
        name = person.find("name").text
        email = person.find("email").text
        print(f"ID: {person_id}, Name: {name}, Email: {email}")
    
    # XPath-like searching
    all_names = root.findall(".//name")  # All name elements
    for name_elem in all_names:
        print(f"Name: {name_elem.text}")


def html_parsing():
    """HTML parsing (requires beautifulsoup4)"""
    try:
        from bs4 import BeautifulSoup
        
        html_content = """
        <html>
            <head><title>Page Title</title></head>
            <body>
                <div class="container">
                    <h1>Hello</h1>
                    <p class="intro">Paragraph 1</p>
                    <p>Paragraph 2</p>
                </div>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Find elements
        title = soup.find("title").text
        h1 = soup.find("h1").text
        
        # Find all
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            print(f"Paragraph: {p.text}")
        
        # CSS selectors
        intro = soup.select_one("p.intro")
        print(f"Intro: {intro.text}")
        
    except ImportError:
        print("beautifulsoup4 not installed; pip install beautifulsoup4")


# ============================================================================
# 7. DIRECTORY OPERATIONS
# ============================================================================

def directory_operations():
    """Create, list, remove directories"""
    base = Path(__file__).parent
    test_dir = base / "test_directory"
    
    # Create directory (mkdir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create nested structure
    (test_dir / "subdir1" / "subdir2").mkdir(parents=True, exist_ok=True)
    
    # List contents
    contents = list(test_dir.iterdir())
    
    # List only files
    files = [p for p in test_dir.rglob("*") if p.is_file()]
    
    # List only directories
    dirs = [p for p in test_dir.rglob("*") if p.is_dir()]
    
    # Remove directory (shutil)
    shutil.rmtree(test_dir)  # Recursive delete
    print(f"Removed {test_dir}")


def walk_directory():
    """Walk directory tree (yields root, dirs, files tuples)"""
    base = Path(__file__).parent
    
    # Using os.walk (traditional)
    for root, dirs, files in os.walk(base):
        level = root.replace(str(base), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{Path(root).name}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:3]:  # Limit output
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")


# ============================================================================
# 8. FILE METADATA
# ============================================================================

def file_metadata():
    """Get file size, modification time, permissions, etc."""
    base = Path(__file__).parent
    test_file = base / "test.txt"
    test_file.write_text("sample content", encoding="utf-8")
    
    # Size in bytes
    size = test_file.stat().st_size
    
    # Modification time (Unix timestamp)
    mtime = test_file.stat().st_mtime
    
    # Access time, change time
    atime = test_file.stat().st_atime
    ctime = test_file.stat().st_ctime
    
    # Convert to datetime
    from datetime import datetime
    mod_datetime = datetime.fromtimestamp(mtime)
    print(f"Modified: {mod_datetime}")
    
    # File mode (permissions)
    mode = test_file.stat().st_mode
    
    # Check readable/writable
    readable = os.access(test_file, os.R_OK)
    writable = os.access(test_file, os.W_OK)
    executable = os.access(test_file, os.X_OK)
    
    print(f"Size: {size} bytes")
    print(f"Readable: {readable}, Writable: {writable}, Executable: {executable}")


# ============================================================================
# 9. TEMPORARY FILES
# ============================================================================

def temporary_files():
    """Create temporary files and directories safely"""
    
    # Named temporary file (persists until closed)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_path = Path(f.name)
        f.write("temporary content")
        print(f"Temp file: {temp_path}")
    
    # Auto-cleanup when context exits
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True) as f:
        f.write("auto-deleted content")
        print(f"Temp file (auto-delete): {f.name}")
    # File deleted here
    
    # Temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        temp_file = temp_dir / "file.txt"
        temp_file.write_text("data")
        print(f"Temp dir: {temp_dir}")
    # Directory and contents deleted here
    
    # Get system temp directory
    system_temp = Path(tempfile.gettempdir())
    print(f"System temp dir: {system_temp}")


# ============================================================================
# 10. LARGE FILE HANDLING
# ============================================================================

def large_file_streaming():
    """Efficiently read large files line-by-line"""
    base = Path(__file__).parent
    large_file = base / "large.txt"
    
    # Create sample large file
    with open(large_file, "w", encoding="utf-8") as f:
        for i in range(1000):
            f.write(f"Line {i}: " + "x" * 100 + "\n")
    
    # LINE BY LINE (memory efficient)
    line_count = 0
    with open(large_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            line_count += 1
            if line_count <= 3:
                print(f"Line: {line[:50]}...")
    
    # CHUNK reading (for binary or custom processing)
    with open(large_file, "rb") as f:
        while True:
            chunk = f.read(8192)  # 8KB chunks
            if not chunk:
                break
            # Process chunk without loading entire file
    
    # GENERATOR for streaming
    def read_lines(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
    
    for i, line in enumerate(read_lines(large_file)):
        if i >= 3:
            break
        print(f"Generated line: {line[:50]}...")


def file_comparison():
    """Compare two files for differences"""
    base = Path(__file__).parent
    file1 = base / "file1.txt"
    file2 = base / "file2.txt"
    
    file1.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    file2.write_text("line 1\nline 2 modified\nline 3\n", encoding="utf-8")
    
    # Simple comparison
    if file1.read_text() == file2.read_text():
        print("Files are identical")
    else:
        print("Files differ")
    
    # Line-by-line diff using difflib
    import difflib
    with open(file1) as f1, open(file2) as f2:
        diff = difflib.unified_diff(
            f1.readlines(),
            f2.readlines(),
            fromfile="file1.txt",
            tofile="file2.txt"
        )
        for line in diff:
            print(line.rstrip())


# ============================================================================
# 11. COMPRESSION
# ============================================================================

def zip_operations():
    """Create and extract ZIP archives"""
    import zipfile
    
    base = Path(__file__).parent
    zip_path = base / "archive.zip"
    extract_dir = base / "extracted"
    
    # Create ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(base / "test.txt", arcname="test.txt")
        zf.write(base / "data.json", arcname="data.json")
    
    # List contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.filelist:
            print(f"File: {info.filename}, Size: {info.file_size}")
    
    # Extract all
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    
    # Extract specific file
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = zf.read("test.txt")  # Returns bytes
        print(f"Extracted: {data}")


def gzip_operations():
    """GZIP compression (single file)"""
    import gzip
    
    base = Path(__file__).parent
    original = base / "test.txt"
    compressed = base / "test.txt.gz"
    
    original.write_text("Hello World! " * 100, encoding="utf-8")
    
    # Compress
    with open(original, "rb") as f_in:
        with gzip.open(compressed, "wb") as f_out:
            f_out.writelines(f_in)
    
    # Decompress
    with gzip.open(compressed, "rb") as f_in:
        decompressed = f_in.read().decode("utf-8")
        print(f"Decompressed: {decompressed[:50]}...")


def tar_operations():
    """TAR archives (with optional compression)"""
    import tarfile
    
    base = Path(__file__).parent
    tar_path = base / "archive.tar.gz"
    
    # Create TAR with gzip compression
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(base / "test.txt", arcname="test.txt")
        tar.add(base / "data.json", arcname="data.json")
    
    # List contents
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"Member: {member.name}, Size: {member.size}")
    
    # Extract all
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(base / "extracted_tar")


# ============================================================================
# 12. FILE LOCKING & CONCURRENCY
# ============================================================================

@contextmanager
def file_lock(filepath, timeout=5):
    """Simple file-based lock (advisory, not enforced by OS)"""
    lock_file = Path(str(filepath) + ".lock")
    import time
    
    start = time.time()
    while lock_file.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Could not acquire lock on {filepath}")
        time.sleep(0.1)
    
    lock_file.touch()
    try:
        yield
    finally:
        lock_file.unlink(missing_ok=True)


def safe_file_write(filepath: Path, content: str, encoding: str = "utf-8"):
    """Atomic write: write to temp file, then rename"""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=filepath.parent,
        delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # Atomic rename (POSIX systems)
    Path(tmp_path).replace(filepath)


def concurrent_file_safety():
    """Safe patterns for concurrent file access"""
    base = Path(__file__).parent
    shared_file = base / "shared.txt"
    
    # Pattern 1: Read-only access (multiple readers OK)
    with open(shared_file, "r") as f:
        content = f.read()
    
    # Pattern 2: Write with lock
    with file_lock(shared_file):
        with open(shared_file, "a") as f:
            f.write("new line\n")
    
    # Pattern 3: Use atomic write
    safe_file_write(shared_file, "atomic content\n")


# ============================================================================
# 13. COMMON UTILITIES
# ============================================================================

def copy_and_move_files():
    """Copy and move files/directories"""
    base = Path(__file__).parent
    source = base / "source.txt"
    dest = base / "dest.txt"
    
    source.write_text("content", encoding="utf-8")
    
    # Copy file
    shutil.copy(source, dest)      # Copy content only
    shutil.copy2(source, dest)     # Copy content + metadata
    
    # Copy directory (recursive)
    shutil.copytree(base / "dir1", base / "dir2")
    
    # Move/rename
    dest.rename(base / "renamed.txt")
    # Or: shutil.move(dest, base / "renamed.txt")


def find_files_by_pattern():
    """Find files matching patterns"""
    base = Path(__file__).parent
    
    # All Python files
    py_files = list(base.glob("*.py"))
    
    # All .txt files recursively
    txt_files = list(base.rglob("*.txt"))
    
    # Multiple patterns
    config_files = list(base.glob("config.*"))
    
    # Using filter with stat
    def find_large_files(directory, min_size_bytes=1024):
        return [
            p for p in directory.rglob("*")
            if p.is_file() and p.stat().st_size >= min_size_bytes
        ]
    
    large = find_large_files(base, min_size_bytes=512)
    print(f"Found {len(large)} large files")


def file_statistics():
    """Analyze files in a directory"""
    from collections import defaultdict
    
    base = Path(__file__).parent
    
    # Count by extension
    extensions = defaultdict(int)
    total_size = 0
    
    for file in base.rglob("*"):
        if file.is_file():
            extensions[file.suffix] += 1
            total_size += file.stat().st_size
    
    print("Files by type:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext if ext else '(no extension)'}: {count}")
    print(f"Total size: {total_size} bytes ({total_size / 1024:.1f} KB)")


@dataclass
class FileInfo:
    """Data class for file metadata"""
    path: Path
    size: int
    is_file: bool
    is_dir: bool
    modified: float
    
    @classmethod
    def from_path(cls, path: Path):
        stat = path.stat()
        return cls(
            path=path,
            size=stat.st_size,
            is_file=path.is_file(),
            is_dir=path.is_dir(),
            modified=stat.st_mtime
        )


def organized_file_listing():
    """Gather file info with dataclass"""
    base = Path(__file__).parent
    
    files_info = [
        FileInfo.from_path(p)
        for p in base.rglob("*")
        if p.is_file()
    ]
    
    # Sort by size descending
    files_info.sort(key=lambda f: f.size, reverse=True)
    
    for info in files_info[:5]:
        print(f"{info.path.name}: {info.size} bytes")


# ============================================================================
# BEST PRACTICES & PATTERNS
# ============================================================================

"""
BEST PRACTICES:

1. Always use pathlib.Path over os.path for modern, readable code

2. Always specify encoding explicitly:
   - UTF-8 is standard: encoding="utf-8"
   - Detect encoding if unknown: use chardet library

3. Use context managers (with statement) for file handles:
   - Ensures files are closed, even on exceptions
   - Use contextlib.contextmanager for custom patterns

4. For large files:
   - Read line-by-line or in chunks, not all at once
   - Use generators for memory efficiency
   - Stream processing patterns

5. When writing files:
   - Write to temp file first, then atomic rename
   - Or use write-then-verify pattern
   - Always handle encode errors: errors="replace" or "ignore"

6. Permissions & security:
   - Use restrictive permissions: mode=0o600 for sensitive files
   - Never use pickle for untrusted data
   - Validate file paths to prevent directory traversal

7. Directory structure:
   - Use mkdir(parents=True, exist_ok=True) for safe creation
   - Clean up temp directories: tempfile.TemporaryDirectory()

8. Error handling:
   - FileNotFoundError, PermissionError, IsADirectoryError
   - Use try/except with specific exception types
   - Handle encoding errors gracefully

9. CSV best practices:
   - Always use newline="" on Windows
   - Use csv module, not manual string splitting
   - Use DictWriter for complex data

10. JSON best practices:
    - Use indent for readable output
    - Set ensure_ascii=False for Unicode
    - Use json.JSONDecodeError for error handling
"""


# ============================================================================
# EXAMPLES: PRACTICAL PATTERNS
# ============================================================================

def example_process_log_file():
    """Common pattern: read log file and extract data"""
    base = Path(__file__).parent
    log_file = base / "app.log"
    
    # Create sample log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("[INFO] App started\n")
        f.write("[ERROR] Connection failed\n")
        f.write("[INFO] Retrying...\n")
        f.write("[ERROR] Still failed\n")
    
    # Extract errors
    errors = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "[ERROR]" in line:
                errors.append(line.strip())
    
    print(f"Found {len(errors)} errors:")
    for error in errors:
        print(f"  {error}")


def example_batch_file_processing():
    """Process all files of a type in directory"""
    import csv as csv_module
    base = Path(__file__).parent
    
    # Create sample CSV files
    for i in range(3):
        csv_path = base / f"data_{i}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=["id", "value"])
            writer.writeheader()
            writer.writerow({"id": i, "value": i * 10})
    
    # Process all CSVs
    total_rows = 0
    for csv_path in base.glob("data_*.csv"):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                total_rows += 1
    
    print(f"Processed {total_rows} total rows")


def example_config_management():
    """Load config from JSON with fallback defaults"""
    base = Path(__file__).parent
    config_file = base / "config.json"
    
    def load_config(path: Path, defaults: dict) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        else:
            path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
            return defaults
    
    defaults = {
        "debug": False,
        "port": 8000,
        "workers": 4
    }
    
    config = load_config(config_file, defaults)
    print(f"Config: {config}")


if __name__ == "__main__":
    print("File Handling Examples\n" + "=" * 50)
    
    # pathlib_basics()
    # text_file_read_write()
    # json_operations()
    # csv_operations()
    # directory_operations()
    # file_metadata()
    # large_file_streaming()
    # zip_operations()
    # example_log_file()
    
    print("Uncomment examples above to run them")
