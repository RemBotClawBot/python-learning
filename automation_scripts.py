"""
automation_scripts.py

Real-world automation helpers showing how to:
- Organize files using pathlib
- Generate status reports
- Schedule reminders with simple text output
- Build a CLI using argparse for flexible execution
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List

WORKSPACE = Path(__file__).parent
BACKUP_DIR = WORKSPACE / "backups"
REPORT_DIR = WORKSPACE / "reports"
BACKUP_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def find_python_files(target: Path) -> List[Path]:
    """Return all Python files under the target directory."""
    return [path for path in target.rglob("*.py") if path.is_file()]


def create_backup(target: Path) -> Path:
    """Zip the target directory into timestamped archive."""
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_path = BACKUP_DIR / f"python-learning-backup-{timestamp}"
    archive = shutil.make_archive(str(archive_path), "zip", root_dir=target)
    return Path(archive)


def generate_repo_report(target: Path) -> Path:
    """Produce JSON summary about Python files (loc, functions, etc.)."""
    summary: Dict[str, Dict[str, int]] = {}
    total_lines = 0

    for file_path in find_python_files(target):
        lines = file_path.read_text(encoding="utf-8").splitlines()
        info = {
            "lines": len(lines),
            "empty": sum(1 for line in lines if not line.strip()),
            "comment": sum(1 for line in lines if line.strip().startswith("#")),
        }
        info["code"] = info["lines"] - info["empty"] - info["comment"]
        summary[str(file_path.relative_to(target))] = info
        total_lines += info["lines"]

    report = {
        "generated_at": dt.datetime.now().isoformat(),
        "total_files": len(summary),
        "total_lines": total_lines,
        "files": summary,
    }

    report_file = REPORT_DIR / "repo_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_file


def schedule_reminder(task: str, minutes_from_now: int) -> str:
    """Return a reminder message (could be sent via cron/notifications)."""
    remind_at = dt.datetime.now() + dt.timedelta(minutes=minutes_from_now)
    return textwrap.dedent(
        f"""
        🔔 Reminder Scheduled
        Task: {task}
        Trigger: {remind_at:%Y-%m-%d %H:%M}
        (Use cron or task scheduler to send this message.)
        """
    ).strip()


def clean_backups(retention: int = 5) -> List[Path]:
    """Keep only the most recent `retention` backup archives."""
    archives = sorted(BACKUP_DIR.glob("python-learning-backup-*.zip"))
    to_remove = archives[:-retention]
    for archive in to_remove:
        archive.unlink()
    return to_remove


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Automation tools for the learning repo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup_parser = subparsers.add_parser("backup", help="Create zip archive of the repo")
    backup_parser.add_argument("--path", type=Path, default=WORKSPACE, help="Target directory")

    report_parser = subparsers.add_parser("report", help="Generate JSON summary of Python files")
    report_parser.add_argument("--path", type=Path, default=WORKSPACE)

    reminder_parser = subparsers.add_parser("reminder", help="Format reminder text")
    reminder_parser.add_argument("task", type=str, help="Reminder description")
    reminder_parser.add_argument("--minutes", type=int, default=60)

    clean_parser = subparsers.add_parser("clean", help="Remove old backup archives")
    clean_parser.add_argument("--retention", type=int, default=5)

    args = parser.parse_args()

    if args.command == "backup":
        archive = create_backup(args.path)
        print(f"✅ Backup created: {archive}")
    elif args.command == "report":
        report_file = generate_repo_report(args.path)
        print(f"✅ Report written to {report_file}")
    elif args.command == "reminder":
        print(schedule_reminder(args.task, minutes_from_now=args.minutes))
    elif args.command == "clean":
        removed = clean_backups(retention=args.retention)
        if removed:
            print("🧹 Removed old archives:")
            for archive in removed:
                print(f"  - {archive}")
        else:
            print("No archives to remove. You're tidy! ✨")


if __name__ == "__main__":
    run_cli()
