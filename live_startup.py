"""Optional producer barrier, after RAW positioning and before replay epoch.

No decoding, inference or clocks live here. The consumer can finish detector
warmup once the SDK reader is prepared, then explicitly release paced replay.
"""


def wait_for_release(connection, start_signal, stop):
    if start_signal is None:
        return not stop.is_set()
    if stop.is_set():
        return False
    connection.send(('prepared', None))
    while not start_signal.wait(.05):
        if stop.is_set():
            return False
    return not stop.is_set()
