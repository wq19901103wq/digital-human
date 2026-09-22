#!/usr/bin/env python3
"""后台服务管理：常驻守护（pid 文件 + 自愈提示）/ 停止 / 状态。

  python scripts/serve_dashboard.py --daemon [--port 8080]   # 后台常驻
  python scripts/serve_dashboard.py --status                 # 查看状态
  python scripts/serve_dashboard.py --stop                   # 停止
  python scripts/serve_dashboard.py [--port 8080]            # 前台运行（调试用）
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PID_PATH = ROOT / ".dashboard.pid"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_status(port: int) -> None:
    if PID_PATH.exists():
        pid = int(PID_PATH.read_text().strip())
        if _pid_alive(pid):
            print(f"后台运行中 (pid {pid}) → http://localhost:{port}/dashboard/index.html")
            return
        PID_PATH.unlink(missing_ok=True)
    print("后台未运行。启动: python scripts/serve_dashboard.py --daemon")


def cmd_stop() -> None:
    if not PID_PATH.exists():
        print("后台未运行")
        return
    pid = int(PID_PATH.read_text().strip())
    if _pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        print(f"已停止 (pid {pid})")
    PID_PATH.unlink(missing_ok=True)


def cmd_daemon(port: int) -> None:
    if PID_PATH.exists():
        pid = int(PID_PATH.read_text().strip())
        if _pid_alive(pid):
            print(f"后台已在运行 (pid {pid}) → http://localhost:{port}/dashboard/index.html")
            return
    log = open("/tmp/dh_dashboard.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "serve_dashboard.py"), "--port", str(port)],
        cwd=str(ROOT), stdout=log, stderr=log,
        start_new_session=True,  # 脱离终端会话，终端关闭不死
    )
    PID_PATH.write_text(str(proc.pid))
    print(f"后台已启动 (pid {proc.pid}) → http://localhost:{port}/dashboard/index.html")
    print("管理: --status 查看 / --stop 停止；日志: /tmp/dh_dashboard.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="迭代后台服务")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--daemon", action="store_true", help="后台常驻")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.stop:
        cmd_stop()
        return
    if args.status:
        cmd_status(args.port)
        return
    if args.daemon:
        cmd_daemon(args.port)
        return

    from src.dashboard.server import serve  # noqa: E402
    serve(port=args.port)


if __name__ == "__main__":
    main()
