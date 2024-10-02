import hashlib
import os


def generate_unique_folder_name(pdf_url: str) -> str:
    # Create a hash of the URL
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
    # Get the last part of the URL as the filename
    original_filename = os.path.basename(pdf_url)
    # Remove the file extension if present
    base_name = os.path.splitext(original_filename)[0]
    # Combine the base name and hash
    return f"{base_name}_{url_hash[:8]}"
