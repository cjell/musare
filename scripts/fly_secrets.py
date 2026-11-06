"""Give the Fly app the secrets it runs on, from .env, without showing any of them.

    ./mscs/python.exe scripts/fly_secrets.py

Only the keys the server reads are sent - not everything in .env, which also
holds keys for scripts and for other services. They go to `flyctl secrets
import` on stdin, so no value is printed or appears on a command line. Staged:
they take effect on the next deploy.

SESSION_SECRET is not in .env. It is generated here the first time, and left
alone afterwards, because changing it signs everyone out.
"""

from __future__ import annotations

import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
APP = "munity"

REQUIRED = [
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "DATABASE_URL",  # capture: enrolment and pulls
    "SUPABASE_URL",  # profile documents and pictures
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",  # charts and edits in plain language
    "TICKETMASTER_API_KEY",  # shows
]
OPTIONAL = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
# Deliberately never sent: SPOTIFY_REDIRECT_URI (fly.toml sets the public one;
# the local 127.0.0.1 value would break sign-in) and DEFAULT_USER (a fallback
# for scripts, which a public server must not have).


def flyctl() -> str:
    found = shutil.which("flyctl") or shutil.which("fly")
    if found:
        return found
    home = Path.home() / ".fly" / "bin" / "flyctl.exe"
    if home.exists():
        return str(home)
    sys.exit("flyctl is not installed - see the deploy steps in the README")


def main() -> int:
    env = dotenv_values(ROOT / ".env")
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        print("missing from .env:", ", ".join(missing))
        return 1
    send = {k: env[k] for k in REQUIRED + OPTIONAL if env.get(k)}

    fly = flyctl()
    listed = subprocess.run(
        [fly, "secrets", "list", "-a", APP], capture_output=True, text=True, check=True
    ).stdout
    if "SESSION_SECRET" not in listed:
        send["SESSION_SECRET"] = secrets.token_urlsafe(48)

    body = "".join(f"{k}={v}\n" for k, v in send.items())
    subprocess.run(
        [fly, "secrets", "import", "-a", APP, "--stage"], input=body, text=True, check=True
    )
    print("staged for the next deploy:", ", ".join(send))
    return 0


if __name__ == "__main__":
    sys.exit(main())
