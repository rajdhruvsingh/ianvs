# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test Case Controller"""

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.common import utils
from core.common.constant import TestObjectType
from core.testcasecontroller.algorithm import Algorithm
from core.testcasecontroller.testcase import TestCase


class TestCaseController:
    """
    Test Case Controller:
    Control the runtime behavior of test cases like instance generation and vanish.
    """

    def __init__(self):
        self.test_cases = []

    def build_testcases(self, test_env, test_object):
        """
        Build multiple test cases by using a test environment and multiple test algorithms.
        """
        test_object_type = test_object.get("type")
        test_object_config = test_object.get(test_object_type)
        if test_object_type == TestObjectType.ALGORITHMS.value:
            algorithms = self._parse_algorithms_config(test_object_config)
            for algorithm in algorithms:
                self.test_cases.append(TestCase(test_env, algorithm))

    def run_testcases(self, workspace, parallel=False, max_workers=None):
        """
        Run all test cases, either sequentially (default) or in parallel.

        Parameters
        ----------
        workspace : str
            The workspace directory for test case outputs.
        parallel : bool or str
            Whether to run test cases in parallel. Accepts bool or YAML string
            values ('true', 'false', '1', 'yes'). Default is False.
        max_workers : int or None
            Maximum number of parallel workers. When None, ThreadPoolExecutor
            uses its own default (min(32, os.cpu_count() + 4)), which is safe
            for resource-intensive ML workloads. Default is None.

        Notes
        -----
        ThreadPoolExecutor is chosen over ProcessPoolExecutor because ML model
        objects (PyTorch, TensorFlow, Sedna paradigms) are often not picklable,
        which is a hard requirement for process-based parallelism.

        Thread-based parallelism provides meaningful speedup for I/O-bound
        workloads (dataset loading, result writing) and GPU-offloaded model
        inference where the GIL is released during compute. For pure CPU-bound
        training workloads, the GIL limits concurrency; in those cases the
        Sandbox Engine (PR #526, Phase 2) with ProcessPoolExecutor will provide
        better isolation and true parallelism.
        """
        # Coerce YAML string values robustly
        if isinstance(parallel, str):
            parallel = parallel.lower() in ("true", "1", "yes")

        if max_workers is not None:
            try:
                max_workers = int(max_workers)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"max_workers must be an integer, got {max_workers!r}"
                ) from err
            if max_workers <= 0:
                raise ValueError(
                    f"max_workers must be greater than 0, got {max_workers}"
                )

        if parallel:
            return self._run_testcases_parallel(workspace, max_workers)
        return self._run_testcases_sequential(workspace)

    def _run_testcases_sequential(self, workspace):
        """
        Run test cases sequentially.

        Preserves the exact original behavior of run_testcases().
        Called when parallel=False (default).
        """
        succeed_results = {}
        succeed_testcases = []
        for testcase in self.test_cases:
            try:
                res, time = (testcase.run(workspace), utils.get_local_time())
            except Exception as err:
                raise RuntimeError(
                    f"testcase(id={testcase.id}) runs failed, error: {err}"
                ) from err
            succeed_results[testcase.id] = (res, time)
            succeed_testcases.append(testcase)
        return succeed_testcases, succeed_results

    def _run_testcases_parallel(self, workspace, max_workers=None):
        """
        Run test cases concurrently using ThreadPoolExecutor.

        Each test case receives a deep copy of its test_env to prevent
        race conditions from shared state mutations (e.g., dataset.load_data()
        or paradigm.run() modifying shared attributes) across threads.

        Failed test cases are collected and reported together rather than
        stopping at the first failure, giving users a complete picture of
        which parameter groups failed.

        Parameters
        ----------
        workspace : str
            The workspace directory for test case outputs.
        max_workers : int or None
            Maximum number of worker threads. When None, ThreadPoolExecutor
            uses min(32, os.cpu_count() + 4) — a safe default for ML workloads
            that avoids OOM from unbounded concurrency.
        """
        succeed_results = {}
        succeed_testcases = []
        failed_testcases = []

        # Deep copy each test case so parallel threads do not share mutable
        # test_env or dataset state. If a test_env or algorithm carries a
        # non-deepcopyable resource (e.g. an open file handle, socket, or
        # GPU/session object attached by a custom paradigm), fail early with
        # a clear, actionable error instead of an opaque TypeError/RuntimeError
        # raised from deep inside copy.deepcopy().
        try:
            parallel_testcases = [copy.deepcopy(tc) for tc in self.test_cases]
        except Exception as err:
            raise RuntimeError(
                "Failed to prepare test cases for parallel execution: "
                f"{type(err).__name__}: {err}. This usually means a test_env "
                "or algorithm object holds a resource that cannot be deep-"
                "copied (e.g. an open file, socket, or GPU/session handle). "
                "Set parallel=False to run sequentially, or ensure custom "
                "paradigms/test_env objects only hold deepcopy-safe state "
                "(config values, paths) rather than live resources."
            ) from err

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_testcase = {
                executor.submit(testcase.run, workspace): testcase
                for testcase in parallel_testcases
            }

            for future in as_completed(future_to_testcase):
                testcase = future_to_testcase[future]
                try:
                    res = future.result()
                    time = utils.get_local_time()
                    succeed_results[testcase.id] = (res, time)
                    succeed_testcases.append(testcase)
                except Exception as err:  # pylint: disable=broad-exception-caught
                    failed_testcases.append((testcase, err))

        if failed_testcases:
            error_msgs = [
                f"  testcase(id={tc.id}) failed: {err}"
                for tc, err in failed_testcases
            ]
            raise RuntimeError(
                f"{len(failed_testcases)} testcase(s) failed in parallel execution:\n"
                + "\n".join(error_msgs)
            )

        return succeed_testcases, succeed_results

    @classmethod
    def _parse_algorithms_config(cls, config):
        algorithms = []
        for algorithm_config in config:
            name = algorithm_config.get("name")
            config_file = algorithm_config.get("url")
            if not utils.is_local_file(config_file):
                raise RuntimeError(
                    f"not found algorithm config file({config_file}) in local"
                )
            try:
                config = utils.yaml2dict(config_file)
                algorithm = Algorithm(name, config)
                algorithms.append(algorithm)
            except Exception as err:
                raise RuntimeError(
                    f"algorithm config file({config_file}) is not supported, "
                    f"error: {err}"
                ) from err

        new_algorithms = []
        for algorithm in algorithms:
            for modules in algorithm.modules_list:
                new_algorithm = copy.deepcopy(algorithm)
                new_algorithm.modules = modules
                new_algorithms.append(new_algorithm)

        return new_algorithms
