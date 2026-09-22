#!/usr/bin/env python3
"""Report completed few-shot ranker AUC and paired selection versus the frozen baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generator.fewshot_ranker.pipeline import status
from src.generator.ranker_report import report
from src.iteration.storage import file_lock, locked, read_json, write_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', required=True, type=Path)
    parser.add_argument('--wait', action='store_true', help='Wait for the current training process to complete')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args(argv)
    output = args.training_output
    if not (output / 'manifest.json').is_file():
        parser.error('training manifest does not exist')
    if read_json(output/'manifest.json').get('kind') == 'cached_fewshot_graded_comparison':
        from src.generator.fewshot_ranker.graded_comparison import report as grade_report
        print(json.dumps(grade_report(output), ensure_ascii=False, indent=2), flush=True)
        return
    if args.background:
        with file_lock(output / '.report_launch.lock', blocking=False):
            if locked(output / '.report_queue.lock'):
                parser.error('this report is already queued')
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       *[v for v in argv if v != '--background']]
            with (output / 'selection_report.log').open('a') as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            write_json(output / 'report_process.json', dict(pid=process.pid, command=command))
            for _ in range(50):
                if locked(output / '.report_queue.lock') or process.poll() is not None:
                    break
                time.sleep(.1)
            print(json.dumps(dict(pid=process.pid, running=process.poll() is None)))
        return
    with file_lock(output / '.report_queue.lock', blocking=False):
        while True:
            current = status(output)
            if not current['running']:
                if current['status'] != 'complete':
                    parser.error('training is not complete: ' + str(current))
                break
            if not args.wait:
                parser.error('training is still running; use --wait')
            time.sleep(15)
        print(json.dumps(report(output), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
