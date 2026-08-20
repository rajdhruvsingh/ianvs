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

"""
Tests for parallel test case execution (PR #532).

Covers:
- Sequential execution (backward compat)
- Parallel execution correctness and isolation
- Output directory uniqueness under concurrency
- YAML string coercion for parallel/max_workers
- max_workers validation
- Partial failure reporting in parallel mode
- Deep copy isolation (shared state not mutated)
"""

import os
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import MagicMock

from core.testcasecontroller.testcase.testcase import TestCase
from core.testcasecontroller.testcasecontroller import TestCaseController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_testcase(result=None, raise_error=None, sleep=0):
    """Return a mock TestCase whose .run() either returns result or raises."""
    tc = MagicMock()
    tc.id = uuid.uuid1()

    if raise_error:
        tc.run.side_effect = raise_error
    elif sleep:
        def _slow_run(workspace):  # pylint: disable=unused-argument
            time.sleep(sleep)
            return result if result is not None else {"metric": 1.0}
        tc.run.side_effect = _slow_run
    else:
        tc.run.return_value = result if result is not None else {"metric": 1.0}

    return tc


def _make_controller(*testcases):
    """Return a TestCaseController pre-loaded with the given mock test cases."""
    ctrl = TestCaseController()
    ctrl.test_cases = list(testcases)
    return ctrl


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------

class TestConstruction(unittest.TestCase):
    """Tests for TestCaseController construction."""

    def test_empty_controller(self):
        """Controller initialises with an empty test_cases list."""
        ctrl = TestCaseController()
        self.assertEqual(ctrl.test_cases, [])


# ---------------------------------------------------------------------------
# 2. Sequential execution (backward compatibility)
# ---------------------------------------------------------------------------

class TestSequentialExecution(unittest.TestCase):
    """Tests for sequential (default) execution path."""

    def test_empty_sequential(self):
        """Empty controller returns empty lists."""
        ctrl = _make_controller()
        testcases, results = ctrl.run_testcases("/tmp/ws", parallel=False)
        self.assertEqual(testcases, [])
        self.assertEqual(results, {})

    def test_single_testcase_sequential(self):
        """Single test case result is stored under the correct id."""
        tc = _make_mock_testcase(result={"f1": 0.9})
        ctrl = _make_controller(tc)
        testcases, results = ctrl.run_testcases("/tmp/ws", parallel=False)
        self.assertEqual(len(testcases), 1)
        self.assertIn(tc.id, results)
        res, _ = results[tc.id]
        self.assertEqual(res, {"f1": 0.9})

    def test_multiple_testcases_sequential_order(self):
        """All test cases run and all results are stored."""
        tcs = [_make_mock_testcase(result={"idx": i}) for i in range(4)]
        ctrl = _make_controller(*tcs)
        testcases, results = ctrl.run_testcases("/tmp/ws", parallel=False)
        self.assertEqual(len(testcases), 4)
        for tc in tcs:
            self.assertIn(tc.id, results)

    def test_sequential_raises_on_failure(self):
        """A failing test case raises RuntimeError in sequential mode."""
        tc = _make_mock_testcase(raise_error=RuntimeError("model failed"))
        ctrl = _make_controller(tc)
        with self.assertRaises(RuntimeError) as ctx:
            ctrl.run_testcases("/tmp/ws", parallel=False)
        self.assertIn("runs failed", str(ctx.exception))

    def test_default_is_sequential(self):
        """parallel defaults to False — calling without kwargs must work."""
        tc = _make_mock_testcase(result={"acc": 0.95})
        ctrl = _make_controller(tc)
        testcases, _ = ctrl.run_testcases("/tmp/ws")
        self.assertEqual(len(testcases), 1)


# ---------------------------------------------------------------------------
# 3. Parallel execution correctness
# ---------------------------------------------------------------------------

class TestParallelExecution(unittest.TestCase):
    """Tests for parallel execution path."""

    def test_empty_parallel(self):
        """Empty controller returns empty lists in parallel mode."""
        ctrl = _make_controller()
        testcases, results = ctrl.run_testcases("/tmp/ws", parallel=True)
        self.assertEqual(testcases, [])
        self.assertEqual(results, {})

    def test_parallel_all_succeed(self):
        """All test cases complete and results are stored."""
        tcs = [_make_mock_testcase(result={"score": i * 0.1}) for i in range(5)]
        ctrl = _make_controller(*tcs)
        testcases, results = ctrl.run_testcases("/tmp/ws", parallel=True)
        self.assertEqual(len(testcases), 5)
        self.assertEqual(len(results), 5)

    def test_parallel_collects_all_failures(self):
        """All failures are reported together, not just the first."""
        tcs = [
            _make_mock_testcase(raise_error=RuntimeError(f"fail-{i}"))
            for i in range(3)
        ]
        ctrl = _make_controller(*tcs)
        with self.assertRaises(RuntimeError) as ctx:
            ctrl.run_testcases("/tmp/ws", parallel=True)
        self.assertIn("3 testcase(s) failed", str(ctx.exception))

    def test_parallel_partial_failure(self):
        """One failing case is reported; the job still raises."""
        good = _make_mock_testcase(result={"ok": True})
        bad = _make_mock_testcase(raise_error=ValueError("bad params"))
        ctrl = _make_controller(good, bad)
        with self.assertRaises(RuntimeError) as ctx:
            ctrl.run_testcases("/tmp/ws", parallel=True)
        self.assertIn("1 testcase(s) failed", str(ctx.exception))

    def test_parallel_max_workers_respected(self):
        """max_workers=1 runs in parallel mode without error."""
        tcs = [_make_mock_testcase() for _ in range(3)]
        ctrl = _make_controller(*tcs)
        testcases, _ = ctrl.run_testcases("/tmp/ws", parallel=True, max_workers=1)
        self.assertEqual(len(testcases), 3)


# ---------------------------------------------------------------------------
# 4. Output directory uniqueness
# ---------------------------------------------------------------------------

class TestOutputDirUniqueness(unittest.TestCase):
    """Tests that concurrent _get_output_dir() calls never collide."""

    def test_concurrent_output_dirs_are_unique(self):
        """20 concurrent threads must produce 20 unique, created directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = []
            lock = threading.Lock()

            def create_dir():
                """Create a unique output directory from a fresh TestCase shell."""
                mock_tc = MagicMock()
                mock_tc.id = uuid.uuid1()
                mock_tc.algorithm.name = "algo"
                real_tc = object.__new__(TestCase)
                real_tc.id = mock_tc.id
                real_tc.algorithm = mock_tc.algorithm
                directory = real_tc._get_output_dir(tmpdir)  # pylint: disable=protected-access
                with lock:
                    dirs.append(directory)

            threads = [threading.Thread(target=create_dir) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(
                len(dirs), len(set(dirs)),
                "Duplicate output directories detected under concurrency"
            )
            for directory in dirs:
                self.assertTrue(
                    os.path.isdir(directory),
                    f"Directory was not created: {directory}"
                )


# ---------------------------------------------------------------------------
# 5. YAML string coercion
# ---------------------------------------------------------------------------

class TestYamlCoercion(unittest.TestCase):
    """Tests that YAML string values for parallel and max_workers are coerced correctly."""

    def test_parallel_string_true(self):
        """'true' string from YAML must be coerced to True."""
        tc = _make_mock_testcase()
        ctrl = _make_controller(tc)
        testcases, _ = ctrl.run_testcases("/tmp/ws", parallel="true")
        self.assertEqual(len(testcases), 1)

    def test_parallel_string_false(self):
        """'false' string from YAML must be coerced to False (sequential)."""
        tc = _make_mock_testcase()
        ctrl = _make_controller(tc)
        testcases, _ = ctrl.run_testcases("/tmp/ws", parallel="false")
        self.assertEqual(len(testcases), 1)

    def test_parallel_string_yes(self):
        """'yes' string from YAML must be coerced to True."""
        tc = _make_mock_testcase()
        ctrl = _make_controller(tc)
        testcases, _ = ctrl.run_testcases("/tmp/ws", parallel="yes")
        self.assertEqual(len(testcases), 1)

    def test_max_workers_string_coercion(self):
        """max_workers='2' from YAML must be coerced to int."""
        tcs = [_make_mock_testcase() for _ in range(3)]
        ctrl = _make_controller(*tcs)
        testcases, _ = ctrl.run_testcases("/tmp/ws", parallel=True, max_workers="2")
        self.assertEqual(len(testcases), 3)


# ---------------------------------------------------------------------------
# 6. max_workers validation
# ---------------------------------------------------------------------------

class TestMaxWorkersValidation(unittest.TestCase):
    """Tests for max_workers boundary and type validation."""

    def test_max_workers_zero_raises(self):
        """max_workers=0 must raise ValueError."""
        ctrl = _make_controller()
        with self.assertRaises(ValueError) as ctx:
            ctrl.run_testcases("/tmp/ws", parallel=True, max_workers=0)
        self.assertIn("greater than 0", str(ctx.exception))

    def test_max_workers_negative_raises(self):
        """max_workers=-1 must raise ValueError."""
        ctrl = _make_controller()
        with self.assertRaises(ValueError) as ctx:
            ctrl.run_testcases("/tmp/ws", parallel=True, max_workers=-1)
        self.assertIn("greater than 0", str(ctx.exception))

    def test_max_workers_invalid_string_raises(self):
        """Non-numeric string must raise ValueError."""
        ctrl = _make_controller()
        with self.assertRaises(ValueError) as ctx:
            ctrl.run_testcases("/tmp/ws", parallel=True, max_workers="banana")
        self.assertIn("integer", str(ctx.exception))

    def test_max_workers_none_is_valid(self):
        """None means ThreadPoolExecutor uses its own safe default."""
        tc = _make_mock_testcase()
        ctrl = _make_controller(tc)
        testcases, _ = ctrl.run_testcases("/tmp/ws", parallel=True, max_workers=None)
        self.assertEqual(len(testcases), 1)


# ---------------------------------------------------------------------------
# 7. Deep copy isolation
# ---------------------------------------------------------------------------

class TestDeepCopyIsolation(unittest.TestCase):
    """Tests that parallel execution does not mutate the original test_cases list."""

    def test_parallel_does_not_mutate_original_testcases(self):
        """Original test_cases list must not be mutated by parallel execution."""
        tc = _make_mock_testcase(result={"acc": 0.8})
        original_id = tc.id
        ctrl = _make_controller(tc)
        ctrl.run_testcases("/tmp/ws", parallel=True)
        self.assertEqual(ctrl.test_cases[0].id, original_id)


if __name__ == "__main__":
    unittest.main()
