"""Spawn target loads only the quota implementation, not unrelated ML fixtures."""
import os


def consume(instance_root, instance, queue):
    os.environ['DH_INSTANCES_ROOT'] = instance_root
    from src.iteration import control, versions
    versions.switch_instance(instance)
    try:
        with control.request():
            queue.put('accepted')
    except control.StopRequested:
        queue.put('stopped')
