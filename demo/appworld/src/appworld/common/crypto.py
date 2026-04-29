import json
import os
from glob import glob
from typing import Any, cast

from appworld.common.constants import S3_BASE_URL, S3_BUCKET_NAME, S3_REGION_NAME
from appworld.common.imports import ensure_package_installed
from appworld.common.io import (
    ensure_file,
    is_lfs_pointer,
    maybe_create_parent_directory,
    write_file,
)
from appworld.common.misc import show_line_diff


def encrypt_data(data: dict[str, object], secret_key: str = "mysecretkey12345") -> str:
    from base64 import b64encode

    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if len(secret_key) != 16:
        raise ValueError("Secret key must be 16 characters long.")
    data_string = json.dumps(data)
    iv = os.urandom(16)
    cipher = Cipher(
        algorithms.AES(secret_key.encode("utf-8")), modes.CBC(iv), backend=default_backend()
    )
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data_string.encode("utf-8")) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    combined = iv + ciphertext
    combined_base64 = b64encode(combined).decode("utf-8")
    return combined_base64


def encryption_key(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    return kdf.derive(password.encode())


def encrypt_bytes(data: bytes, password: str, salt: bytes) -> bytes:
    import secrets

    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key = encryption_key(password, salt)
    iv = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()
    return iv + encrypted_data


def decrypt_bytes(data: bytes, password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key = encryption_key(password, salt)
    iv = data[:16]
    encrypted_data = data[16:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    return decrypted_data


def pack_bundle(
    bundle_file_path: str,
    base_directory: str,
    include_directories: list[str],
    password: str,
    salt: bytes,
    include_file_path_substrings: list[str] | None = None,
    exclude_extensions: list[str] | None = None,
    include_license: bool = True,
) -> list[str]:
    import io
    import zipfile

    exception_file_names = ["LICENSE", "README_BEFORE_SHARING.md"]
    include_file_path_substrings = include_file_path_substrings or []
    exclude_extensions = exclude_extensions or []
    if include_license:
        for include_directory in include_directories:
            license_file_path = os.path.join(base_directory, include_directory, "LICENSE")
            license_string = (
                "The contents of this folder are part of the protected portion "
                "of AppWorld, released as an encrypted .bundle file under the "
                "Apache 2.0 license with the additional requirement that any "
                "public redistribution of it (or of its derivatives) must also "
                "be done in an encrypted format."
            )
            write_file(license_string, license_file_path)
            readme_file_path = os.path.join(
                base_directory, include_directory, "README_BEFORE_SHARING.md"
            )
            readme_string = (
                "# License\n\n"
                + license_string
                + "\n\n"
                + "Visit https://github.com/stonybrooknlp/appworld for more information."
            )
            write_file(readme_string, readme_file_path)
    bytes_io = io.BytesIO()
    exclude_extensions = [*exclude_extensions, ".bundle"]
    exclude_extensions_ = tuple("." + ext.lower().lstrip(".") for ext in exclude_extensions)
    ignore_directory_names = [
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        ".history",
        ".vscode",
    ]
    packed_file_paths: list[str] = []
    with zipfile.ZipFile(bytes_io, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in glob(base_directory + "/**/*", recursive=True):
            relative_directory = os.path.relpath(os.path.dirname(file_path), base_directory)

            def wrap(x: str) -> str:
                return os.sep + x.strip(os.sep) + os.sep

            if all(
                wrap(directory_) not in wrap(relative_directory)
                for directory_ in include_directories
            ):
                continue
            is_not_exception = all(
                exception_file_name not in file_path for exception_file_name in exception_file_names
            )
            if is_not_exception and os.path.isdir(file_path):
                # NOTE: This should only happend after filter by include_directories.
                continue
            file_name = os.path.basename(file_path)
            if is_not_exception and file_name.endswith(exclude_extensions_):
                continue
            relative_file_path = os.path.relpath(file_path, base_directory)
            if (
                is_not_exception
                and include_file_path_substrings
                and not any(
                    substring in relative_file_path for substring in include_file_path_substrings
                )
            ):
                continue
            if any(
                sub_path in ignore_directory_names
                for sub_path in os.path.abspath(file_path).split(os.path.sep)[:-1]
            ):
                continue
            arcname = os.path.relpath(file_path, base_directory)
            zipf.write(file_path, arcname)
            assert isinstance(arcname, str)
            packed_file_paths.append(arcname)
    zip_data = bytes_io.getvalue()
    encrypted_data = encrypt_bytes(zip_data, password, salt)
    maybe_create_parent_directory(bundle_file_path)
    write_file(encrypted_data, bundle_file_path, "wb")
    return packed_file_paths


def unpack_bundle(
    bundle_file_path: str,
    base_directory: str,
    password: str,
    salt: bytes,
) -> list[str]:
    import io
    import zipfile

    if not os.path.exists(bundle_file_path):
        raise Exception(f"File {bundle_file_path} does not exist.")
    if not os.path.isfile(bundle_file_path):
        raise Exception(f"File {bundle_file_path} is not a file.")
    if is_lfs_pointer(bundle_file_path):
        raise Exception(f"File {bundle_file_path} is a Git LFS pointer and not a bundle file.")
    with open(bundle_file_path, "rb") as file:
        encrypted_data = file.read()
    zip_data = decrypt_bytes(encrypted_data, password, salt)
    bytes_io = io.BytesIO(zip_data)
    with zipfile.ZipFile(bytes_io, "r") as file:
        relative_unpacked_file_paths: list[str] = file.namelist()
        file.extractall(base_directory)
    return relative_unpacked_file_paths


def bundle_file_list(bundle_file_path: str, password: str, salt: bytes) -> list[str]:
    import io
    import zipfile

    ensure_file(bundle_file_path)
    with open(bundle_file_path, "rb") as f:
        encrypted_data = f.read()
    zip_data = decrypt_bytes(encrypted_data, password, salt)
    bytes_io = io.BytesIO(zip_data)
    file_paths = []
    with zipfile.ZipFile(bytes_io, "r") as zipf:
        for file_info in zipf.infolist():
            file_paths.append(file_info.filename)
    sorted_file_paths = sorted(file_paths)
    return sorted_file_paths


def bundle_file_path_to_content(
    bundle_file_path: str,
    password: str,
    salt: bytes,
    include_file_paths: list[str] | None = None,
) -> dict[str, str]:
    import io
    import zipfile

    ensure_file(bundle_file_path)
    if not os.path.exists(bundle_file_path):
        raise Exception(f"File {bundle_file_path} does not exist.")
    with open(bundle_file_path, "rb") as f:
        encrypted_data = f.read()
    zip_data = decrypt_bytes(encrypted_data, password, salt)
    bytes_io = io.BytesIO(zip_data)
    file_path_to_content: dict[str, str] = {}
    with zipfile.ZipFile(bytes_io, "r") as zipf:
        for file_info in zipf.infolist():
            with zipf.open(file_info) as file:
                file_path = file_info.filename
                if include_file_paths is not None and file_path not in include_file_paths:
                    continue
                content = file.read().decode("utf-8")
                file_path_to_content[file_path] = content
    return file_path_to_content


def bundle_file_content(
    bundle_file_path: str,
    password: str,
    salt: bytes,
    file_path: str,
) -> str:
    ensure_file(bundle_file_path)
    file_path_to_content = bundle_file_path_to_content(
        bundle_file_path=bundle_file_path,
        password=password,
        salt=salt,
        include_file_paths=[file_path],
    )
    if file_path not in file_path_to_content:
        raise Exception(f"File {file_path} not found in bundle.")
    content = file_path_to_content[file_path]
    return content


def bundles_file_content_diff(
    bundle_file_path_1: str,
    bundle_file_path_2: str,
    file_path: str,
    password: str,
    salt: bytes,
) -> None:
    file_path_to_content_1 = bundle_file_path_to_content(bundle_file_path_1, password, salt)
    file_path_to_content_2 = bundle_file_path_to_content(bundle_file_path_2, password, salt)
    if file_path not in file_path_to_content_1:
        raise Exception(f"File {file_path} not found in bundle 1.")
    if file_path not in file_path_to_content_2:
        raise Exception(f"File {file_path} not found in bundle 2.")
    content_1 = file_path_to_content_1[file_path]
    content_2 = file_path_to_content_2[file_path]
    show_line_diff(content_1, content_2, show_unchanged=False)


def bundles_file_list_diff(
    bundle_file_path_1: str,
    bundle_file_path_2: str,
    password: str,
    salt: bytes,
) -> dict[str, list[str]]:
    file_path_to_content_1 = bundle_file_path_to_content(bundle_file_path_1, password, salt)
    file_path_to_content_2 = bundle_file_path_to_content(bundle_file_path_2, password, salt)
    file_paths_1 = set(file_path_to_content_1.keys())
    file_paths_2 = set(file_path_to_content_2.keys())
    file_paths_in_both = file_paths_1.intersection(file_paths_2)
    file_paths_only_in_1 = file_paths_1 - file_paths_2
    file_paths_only_in_2 = file_paths_2 - file_paths_1
    file_paths_in_both_with_different_contents = sorted(
        [
            file_path
            for file_path in file_paths_in_both
            if file_path_to_content_1[file_path] != file_path_to_content_2[file_path]
        ]
    )
    return {
        "only_in_1": sorted(file_paths_only_in_1),
        "only_in_2": sorted(file_paths_only_in_2),
        "in_both_but_different": file_paths_in_both_with_different_contents,
    }


def upload_bundle(bundle_file_path: str) -> None:
    ensure_package_installed("boto3")
    import boto3

    s3 = cast(Any, boto3).client("s3", region_name=S3_REGION_NAME)
    bundle_name = os.path.basename(bundle_file_path)
    s3.upload_file(bundle_file_path, S3_BUCKET_NAME, bundle_name, ExtraArgs={"ACL": "public-read"})
    print(f"Uploaded to public URL: {S3_BASE_URL + bundle_name}")
