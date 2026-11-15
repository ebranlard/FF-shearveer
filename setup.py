
#!/usr/bin/env python3

import platform
import shutil
import subprocess
from pathlib import Path


# ============================================================
# Repositories
# ============================================================

WORKSPACE = Path(__file__).resolve().parent
REPOS = WORKSPACE / "repos"

REPOSITORIES = {
    "openfast": {
        "url": "https://github.com/OpenFAST/openfast.git",
        "branch": "dev",
    },
    "r-test": {
        "url": "https://github.com/OpenFAST/r-test.git",
        "branch": "dev",
    },
    "FF-shearveer": {
        "url": "https://github.com/jjmoraa/openfast.git",
        "branch": "FF-shearveer",
    },
}


# ============================================================
# Utilities
# ============================================================

def run(command, cwd=None):
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def check_requirements():

    print("Operating system:", platform.system())
    print("Architecture:", platform.machine())

    if shutil.which("git") is None:
        raise RuntimeError(
            "Git is not installed or not available in PATH."
        )

    print("Git:", shutil.which("git"))

    if shutil.which("cmake"):
        print("CMake:", shutil.which("cmake"))
    else:
        print("WARNING: CMake not found.")


def clone_or_update(name, config):

    destination = REPOS / name

    print(f"\n{'=' * 60}")
    print(f"Preparing {name}")
    print(f"{'=' * 60}")

    if not destination.exists():

        print("Cloning repository...")

        run([
            "git",
            "clone",
            "--branch", config["branch"],
            config["url"],
            str(destination),
        ])

        return

    if not (destination / ".git").exists():
        raise RuntimeError(
            f"{destination} exists but is not a Git repository."
        )

    current_url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"],
        cwd=destination,
        text=True,
    ).strip()

    if current_url != config["url"]:
        raise RuntimeError(
            f"Unexpected repository URL in {destination}"
        )

    # Do not overwrite local modifications.
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=destination,
        text=True,
    )

    if status.strip():
        print("Local changes detected.")
        print("Skipping update to protect your modifications.")
        return

    print("Fetching updates...")

    run(["git", "fetch", "origin"], cwd=destination)

    run(
        ["git", "switch", config["branch"]],
        cwd=destination,
    )

    print("Updating branch...")

    run(
        ["git", "pull", "--ff-only", "origin", config["branch"]],
        cwd=destination,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\nOpenFAST Development Environment")
    print("=" * 60)

    check_requirements()

    REPOS.mkdir(parents=True, exist_ok=True)

    for name, config in REPOSITORIES.items():
        clone_or_update(name, config)

    print("\n" + "=" * 60)
    print("WORKSPACE SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
