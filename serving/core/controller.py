import re
import threading
from collections import deque
from contextlib import nullcontext
from .logger import get_logger

class Controller():
    def __init__(self, total_num, host_timing=None):
        self.end_dict = {}
        self.total_num = total_num
        self.host_timing = host_timing
        self.logger = get_logger(self.__class__)
        self._drain_thread = None
        self._drained_lines = deque(maxlen=256)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        measurement = (
            self.host_timing.measure("astra_wait")
            if self.host_timing is not None else nullcontext()
        )
        with measurement:
            out = [""]
            while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
                line = p.stdout.readline()
                if line == "":
                    return_code = p.poll()
                    detail = (
                        f" with exit code {return_code}"
                        if return_code is not None else " unexpectedly"
                    )
                    raise RuntimeError(
                        f"ASTRA-Sim stdout closed{detail} while waiting "
                        "for the next workload command.")
                # For debugging
                # print(line, end='')
                out.append(line)
                p.stdout.flush()
        return out

    def check_end(self, p):
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=10)
            terminal = next((
                line for line in reversed(self._drained_lines)
                if line in (
                    "All Request Has Been Exited\n",
                    "ERROR: Some Requests Remain\n",
                )
            ), None)
            if terminal is None:
                raise RuntimeError(
                    "ASTRA-Sim stdout ended without a completion summary.")
            print("Checking Non-Exited Systems ...")
            print(terminal, end='')
            return list(self._drained_lines)
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def start_drain(self, p):
        """Continuously drain legacy stdout after IPC completion takes over."""

        if self._drain_thread is not None:
            return

        def drain():
            for line in p.stdout:
                self._drained_lines.append(line)

        self._drain_thread = threading.Thread(
            target=drain, name="astra-stdout-drain", daemon=True)
        self._drain_thread.start()

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return
