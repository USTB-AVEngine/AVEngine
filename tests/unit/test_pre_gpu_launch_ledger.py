from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY / "tools/qa"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import pre_gpu_launch_ledger as LEDGER  # noqa: E402


class PreGpuLaunchLedgerTests(unittest.TestCase):
    def _prepared(self, root: Path) -> tuple[Path, Path, LEDGER.PreparedAttemptSpec]:
        attempt = root / "diagnostic_attempt_01"
        archive = root / "diagnostic_prepare_archive_01"
        capture = root / "capture_attempt_01"
        attempt.mkdir(parents=True)
        request = {
            "schema": "example_request_v1",
            "status": "prepared_not_launched",
            "episode_id": "episode-1",
            "attempt_root": str(attempt),
            "capture_output": str(capture),
        }
        (attempt / "request.json").write_text(
            json.dumps(request) + "\n", encoding="utf-8"
        )
        spec = LEDGER.PreparedAttemptSpec(
            request_schema="example_request_v1",
            request_keys=frozenset(request),
            workspace_roots=(root,),
            expected_fields={"episode_id": "episode-1"},
            expected_paths={
                "attempt_root": attempt,
                "capture_output": capture,
            },
            forbidden_paths=(capture,),
        )
        return attempt, archive, spec

    def test_archive_preserves_request_and_binds_both_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt, archive, spec = self._prepared(Path(directory))
            receipt_path = LEDGER.archive_prepared_attempt(
                attempt_root=attempt,
                archive_root=archive,
                spec=spec,
                reason="request commit became stale",
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(attempt.exists())
            self.assertEqual(receipt_path, archive / spec.receipt_filename)
            self.assertEqual(receipt["original_attempt_root"], str(attempt))
            self.assertEqual(receipt["archive_root"], str(archive))
            self.assertEqual(
                receipt["original_request_path"], str(attempt / "request.json")
            )
            self.assertEqual(
                receipt["archived_request_path"], str(archive / "request.json")
            )
            self.assertFalse(receipt["capture_launch_gpu_query_started"])
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertTrue((archive / "request.json").is_file())
            reopened = LEDGER.verify_preparation_archive(
                archive_root=archive,
                original_attempt_root=attempt,
                spec=spec,
            )
            self.assertEqual(
                reopened["archived_request_path"], str(archive / "request.json")
            )
            attempt.mkdir()
            fresh_request = {
                "schema": "fresh_request_v2",
                "status": "prepared_not_launched",
            }
            (attempt / "request.json").write_text(
                json.dumps(fresh_request) + "\n", encoding="utf-8"
            )
            reopened_after_reuse = LEDGER.verify_preparation_archive(
                archive_root=archive,
                original_attempt_root=attempt,
                spec=spec,
            )
            self.assertEqual(
                reopened_after_reuse["archived_request_path"],
                str(archive / "request.json"),
            )

    def test_verify_returns_exact_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt, _, spec = self._prepared(Path(directory))
            request = LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)
            self.assertEqual(request["episode_id"], "episode-1")

    def test_rejects_wrong_schema_status_and_extra_key(self) -> None:
        for mutation, message in (
            ({"schema": "wrong"}, "schema drift"),
            ({"status": "dry_run_pass_not_launched"}, "status drift"),
            ({"unexpected": True}, "keys do not match"),
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                attempt, _, spec = self._prepared(Path(directory))
                path = attempt / "request.json"
                request = json.loads(path.read_text(encoding="utf-8"))
                request.update(mutation)
                path.write_text(json.dumps(request) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(LEDGER.PreGpuLaunchLedgerError, message):
                    LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)

    def test_rejects_scalar_and_path_drift(self) -> None:
        for key, value, message in (
            ("episode_id", "episode-2", "field drift"),
            ("attempt_root", "/wrong/attempt", "path drift"),
            ("capture_output", "/wrong/capture", "path drift"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                attempt, _, spec = self._prepared(Path(directory))
                path = attempt / "request.json"
                request = json.loads(path.read_text(encoding="utf-8"))
                request[key] = value
                path.write_text(json.dumps(request) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(LEDGER.PreGpuLaunchLedgerError, message):
                    LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)

    def test_rejects_dry_running_final_and_logs(self) -> None:
        names = (
            "dry_run_receipt.json",
            "running_receipt.json",
            "final_receipt.json",
            "capture_stdout.log",
            "capture_stderr.log",
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                attempt, _, spec = self._prepared(Path(directory))
                (attempt / name).write_text("evidence\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    LEDGER.PreGpuLaunchLedgerError, "launch evidence"
                ):
                    LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)

    def test_rejects_capture_output_and_unexpected_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt, _, spec = self._prepared(Path(directory))
            Path(spec.expected_paths["capture_output"]).mkdir()
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "capture evidence"
            ):
                LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)
        with tempfile.TemporaryDirectory() as directory:
            attempt, _, spec = self._prepared(Path(directory))
            (attempt / "notes.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "entries are not closed"
            ):
                LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)

    def test_rejects_attempt_request_entry_and_external_broken_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, _, spec = self._prepared(root)
            real_attempt = root / "real_attempt"
            attempt.rename(real_attempt)
            attempt.symlink_to(real_attempt, target_is_directory=True)
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError,
                "prepared attempt directory must not be a symlink",
            ):
                LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)
        with tempfile.TemporaryDirectory() as directory:
            attempt, _, spec = self._prepared(Path(directory))
            (attempt / "request.json").unlink()
            (attempt / "request.json").symlink_to("missing.json")
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "contains symlinks"
            ):
                LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)
        with tempfile.TemporaryDirectory() as directory:
            attempt, _, spec = self._prepared(Path(directory))
            capture = Path(spec.expected_paths["capture_output"])
            capture.symlink_to("missing-capture", target_is_directory=True)
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "capture evidence"
            ):
                LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=spec)

    def test_rejects_existing_archive_and_non_sibling_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt, archive, spec = self._prepared(Path(directory))
            archive.mkdir()
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "destination already exists"
            ):
                LEDGER.archive_prepared_attempt(
                    attempt_root=attempt,
                    archive_root=archive,
                    spec=spec,
                    reason="stale",
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, _, spec = self._prepared(root)
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "must be a sibling"
            ):
                LEDGER.archive_prepared_attempt(
                    attempt_root=attempt,
                    archive_root=root / "elsewhere/archive",
                    spec=spec,
                    reason="stale",
                )

    def test_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            attempt, _, spec = self._prepared(root)
            escaped_spec = LEDGER.PreparedAttemptSpec(
                request_schema=spec.request_schema,
                request_keys=spec.request_keys,
                workspace_roots=(workspace,),
                expected_fields=spec.expected_fields,
                expected_paths=spec.expected_paths,
                forbidden_paths=spec.forbidden_paths,
            )
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "escapes declared workspace"
            ):
                LEDGER.verify_prepared_attempt(attempt_root=attempt, spec=escaped_spec)

    def test_atomic_publication_race_is_no_replace_and_keeps_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt, archive, spec = self._prepared(Path(directory))
            original_publish = LEDGER.atomic_publish_directory

            def race(*args: object, **kwargs: object) -> Path:
                archive.mkdir()
                return original_publish(*args, **kwargs)

            with (
                mock.patch.object(LEDGER, "atomic_publish_directory", side_effect=race),
                self.assertRaisesRegex(
                    LEDGER.PreGpuLaunchLedgerError,
                    "atomic archive publication failed",
                ),
            ):
                LEDGER.archive_prepared_attempt(
                    attempt_root=attempt,
                    archive_root=archive,
                    spec=spec,
                    reason="stale",
                )
            self.assertTrue((attempt / spec.request_filename).is_file())
            receipt = json.loads(
                (attempt / spec.receipt_filename).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], LEDGER.ARCHIVE_RECEIPT_STATUS)
            self.assertFalse(receipt["attempt_consumed"])

    def test_archive_verifier_rejects_receipt_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt, archive, spec = self._prepared(Path(directory))
            receipt_path = LEDGER.archive_prepared_attempt(
                attempt_root=attempt,
                archive_root=archive,
                spec=spec,
                reason="stale",
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["attempt_consumed"] = True
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LEDGER.PreGpuLaunchLedgerError, "attempt_consumed"
            ):
                LEDGER.verify_preparation_archive(
                    archive_root=archive,
                    original_attempt_root=attempt,
                    spec=spec,
                )


if __name__ == "__main__":
    unittest.main()
