from __future__ import annotations

from contextlib import redirect_stderr
import errno
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.train import atomic_save


class CheckpointIOTests(unittest.TestCase):
    def test_save_replaces_checkpoint_with_loadable_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save({"step": 115500}, path)
            atomic_save({"step": 116000, "weights": torch.arange(3)}, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["step"], 116000)
            torch.testing.assert_close(loaded["weights"], torch.arange(3))
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_partial_write_is_retried_without_replacing_previous_checkpoint(self):
        failures = (
            OSError(errno.EIO, "Input/output error"),
            RuntimeError("basic_ios::clear: iostream error"),
            RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed"),
            RuntimeError("unexpected pos 64 vs 0"),
        )
        for failure in failures:
            with self.subTest(error=failure), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pt"
                torch.save({"step": 115500}, path)
                previous = path.read_bytes()
                original_save = torch.save
                calls = 0

                def flaky_save(payload, stream):
                    nonlocal calls
                    calls += 1
                    self.assertEqual(path.read_bytes(), previous)
                    if calls == 1:
                        stream.write(b"incomplete archive")
                        raise failure
                    original_save(payload, stream)

                with patch("src.train.torch.save", side_effect=flaky_save), \
                     patch("src.train.time.sleep") as sleep, redirect_stderr(io.StringIO()):
                    atomic_save({"step": 116000}, path)
                self.assertEqual(calls, 2)
                sleep.assert_called_once_with(1)
                self.assertEqual(torch.load(path, weights_only=True)["step"], 116000)
                self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_persistent_write_failure_preserves_checkpoint_and_reports_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save({"step": 115500}, path)
            previous = path.read_bytes()

            def disk_full(payload, stream):
                stream.write(b"incomplete archive")
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch("src.train.torch.save", side_effect=disk_full) as save, \
                 patch("src.train.time.sleep") as sleep, redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Checkpoint save failed \\(3/3\\)") as caught:
                    atomic_save({"step": 116000}, path)
            self.assertEqual(save.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(caught.exception.__cause__.errno, errno.ENOSPC)
            self.assertIn("quota", str(caught.exception))
            self.assertIn("free space", str(caught.exception))
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 115500)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_flush_and_rename_failures_preserve_checkpoint(self):
        for target in ("src.train.os.fsync", "pathlib.Path.replace"):
            with self.subTest(stage=target), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pt"
                torch.save({"step": 115500}, path)
                previous = path.read_bytes()
                with patch(target, side_effect=OSError(errno.EIO, "Input/output error")), \
                     patch("src.train.time.sleep"), redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "previous checkpoint has not been replaced"):
                        atomic_save({"step": 116000}, path)
                self.assertEqual(path.read_bytes(), previous)
                self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_storage_query_failure_does_not_hide_write_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            failure = OSError(errno.EIO, "Input/output error")
            with patch("src.train.torch.save", side_effect=failure), \
                 patch("src.train.shutil.disk_usage", side_effect=OSError("mount unavailable")), \
                 patch("src.train.time.sleep"), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Could not query filesystem") as caught:
                    atomic_save({"step": 116000}, path)
            self.assertIs(caught.exception.__cause__, failure)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_unrelated_serialization_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save({"step": 115500}, path)
            failure = RuntimeError("unrelated tensor serialization failure")
            with patch("src.train.torch.save", side_effect=failure) as save, \
                 patch("src.train.time.sleep") as sleep:
                with self.assertRaises(RuntimeError) as caught:
                    atomic_save({"step": 116000}, path)
            self.assertIs(caught.exception, failure)
            save.assert_called_once()
            sleep.assert_not_called()
            self.assertEqual(torch.load(path, weights_only=True)["step"], 115500)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
