"""Single-owner credentials for the private dashboard, stored outside served files."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import secrets
import stat


def credentials_path() -> Path:
    return Path.home() / '.config' / 'digital-human' / 'dashboard-auth.json'


def read_credentials(path: Path) -> dict[str, str]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Dashboard credentials must be an owner-only regular file')
    value = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or not isinstance(value.get('username'), str)
            or not value['username'] or ':' in value['username']
            or not isinstance(value.get('password'), str) or len(value['password']) < 32):
        raise ValueError('Invalid dashboard credentials')
    return value


def initialize(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        read_credentials(path)
        return
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump({'username': 'owner', 'password': secrets.token_urlsafe(32)}, stream)
        stream.write('\n')


def authorization(path: Path) -> str:
    value = read_credentials(path)
    token = base64.b64encode(f"{value['username']}:{value['password']}".encode()).decode('ascii')
    return 'Basic ' + token
