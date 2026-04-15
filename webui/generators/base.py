"""Base generator class for video and image generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import subprocess


class BaseGenerator(ABC):
    """Base class for all generators."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> Tuple[bool, str]:
        """Execute generation.

        Returns:
            Tuple of (success, error_message)
        """
        pass

    def run_command(
        self,
        cmd: list,
        cwd: str = None,
        capture_output: bool = True,
    ) -> subprocess.Popen:
        """Run a command as a subprocess.

        Args:
            cmd: Command and arguments as a list.
            cwd: Working directory.
            capture_output: Whether to capture stdout/stderr.

        Returns:
            Popen object.
        """
        if capture_output:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
            )
        else:
            return subprocess.Popen(cmd, cwd=cwd)

    def wait_for_process(self, process: subprocess.Popen) -> int:
        """Wait for process to complete and consume output.

        Args:
            process: Popen object.

        Returns:
            Return code.
        """
        # Consume output to prevent buffer overflow
        for _ in process.stdout:
            pass
        return process.wait()
