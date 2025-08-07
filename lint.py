#!/usr/bin/env python3
"""
Comprehensive Python code quality checker and formatter.

This script runs all the linting tools used in the CI pipeline:
- Black (code formatting)
- isort (import sorting)
- mypy (type checking)
- bandit (security linting)

Usage:
    python lint.py          # Check and fix formatting issues
    python lint.py --check  # Check only (no fixes)
"""

import subprocess
import sys
from typing import List, Tuple


def run_command(cmd: List[str], description: str, fix_mode: bool = True) -> Tuple[int, str, str]:
    """Run a command and return results."""
    print(f"\n{'='*50}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        return 1, "", "Command timed out"
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return 1, "", str(e)


def main():
    """Main linting function."""
    check_only = "--check" in sys.argv

    print("Python Code Quality Checker")
    print(f"Mode: {'Check only' if check_only else 'Fix and check'}")

    exit_code = 0

    # 1. Black - Code formatting
    if check_only:
        black_cmd = ["black", "--check", "--diff", "."]
        black_desc = "Black (code format check)"
    else:
        black_cmd = ["black", "."]
        black_desc = "Black (code formatting)"

    code, stdout, stderr = run_command(black_cmd, black_desc, not check_only)
    if code != 0:
        exit_code = 1
        if check_only:
            print("❌ Black: Code formatting issues found")
        else:
            print("❌ Black: Failed to format code")
    else:
        print("✅ Black: Code formatting OK")

    # 2. isort - Import sorting
    if check_only:
        isort_cmd = ["isort", "--check-only", "--diff", "."]
        isort_desc = "isort (import sort check)"
    else:
        isort_cmd = ["isort", "."]
        isort_desc = "isort (import sorting)"

    code, stdout, stderr = run_command(isort_cmd, isort_desc, not check_only)
    if code != 0:
        exit_code = 1
        if check_only:
            print("❌ isort: Import sorting issues found")
        else:
            print("❌ isort: Failed to sort imports")
    else:
        print("✅ isort: Import sorting OK")

    # 3. mypy - Type checking (informational only)
    mypy_cmd = ["mypy", "main.py", "--ignore-missing-imports"]
    code, stdout, stderr = run_command(mypy_cmd, "mypy (type checking)", False)
    if code != 0:
        print("⚠️  mypy: Type checking issues found (informational only)")
    else:
        print("✅ mypy: Type checking OK")

    # 4. bandit - Security linting (informational only)
    bandit_cmd = ["bandit", "-r", ".", "-f", "json", "-o", "bandit-report.json"]
    code, stdout, stderr = run_command(bandit_cmd, "bandit (security check)", False)
    if code != 0:
        print("⚠️  bandit: Security issues found (informational only)")
        print("Check bandit-report.json for details")
    else:
        print("✅ bandit: Security check OK")

    # Final summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print("=" * 50)

    if exit_code == 0:
        print("🎉 All linting checks passed!")
        if not check_only:
            print("Code has been formatted and is ready for commit.")
    else:
        print("❌ Some linting checks failed.")
        if check_only:
            print("Run 'python lint.py' (without --check) to fix formatting issues.")
        else:
            print("Please fix the remaining issues manually.")

    print(f"\nMode used: {'Check only' if check_only else 'Fix and check'}")
    print("Available modes:")
    print("  python lint.py          # Fix formatting issues")
    print("  python lint.py --check  # Check only (CI mode)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
