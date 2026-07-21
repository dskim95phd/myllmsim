import os
import importlib
import subprocess
import sys
import types
from functools import lru_cache
from time import time
from .request import *
from .logger import get_logger
from .host_timing import timed_stage
from .run_paths import input_path

logger = get_logger("GraphGenerator")


@lru_cache(maxsize=None)
def _load_llm_converter(chakra_root):
    """Load the repository's Chakra converter once in the serving process."""
    chakra_root = os.path.abspath(chakra_root)
    try:
        module = importlib.import_module("chakra.src.converter.llm_converter")
    except ModuleNotFoundError as error:
        if error.name != "chakra":
            raise
        package_paths = {
            "chakra": chakra_root,
            "chakra.src": os.path.join(chakra_root, "src"),
            "chakra.schema": os.path.join(chakra_root, "schema"),
        }
        for package_name, package_path in package_paths.items():
            package = types.ModuleType(package_name)
            package.__path__ = [package_path]
            sys.modules[package_name] = package
        module = importlib.import_module("chakra.src.converter.llm_converter")
    return module.LLMConverter


def _convert_in_process(chakra_root, trace_path, output_path, num_npus,
                        npu_offset, enable_local_offloading):
    converter_class = _load_llm_converter(chakra_root)
    converter = converter_class(
        trace_path,
        output_path,
        num_npus,
        npu_offset,
        enable_local_offloading,
    )
    converter.convert()


@timed_stage("chakra_conversion")
def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0,
                   npu_offset=0, enable_local_offloading=False, event=False,
                   workload_name=None, inputs_root=None, cleanup_trace=True,
                   converter_mode="in-process"):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    if converter_mode == "in-process":
        logger.debug(
            "Generating graph with the in-process Chakra converter.",
            extra={"node_id": node_id, "instance_id": instance_id},
        )
        _convert_in_process(
            chakra,
            trace_path,
            output_path,
            num_npus,
            npu_offset,
            enable_local_offloading,
        )
    elif converter_mode == "subprocess":
        cmd = [
            sys.executable, '-m', 'chakra.src.converter.converter', 'LLM',
            '--input', trace_path,
            '--output', output_path,
            '--num-npus', str(num_npus),
            '--npu-offset', str(npu_offset),
        ]

        if enable_local_offloading:
            cmd.append('--local-offloading')

        logger.debug(
            "Generating graph with command: %s", " ".join(cmd),
            extra={"node_id": node_id, "instance_id": instance_id},
        )
        subprocess.run(cmd, cwd=chakra, text=True, check=True)
    else:
        raise ValueError(f"Unsupported Chakra converter mode: {converter_mode}")
    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return
