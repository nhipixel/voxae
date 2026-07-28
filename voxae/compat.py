"""Standard-library shims for older interpreters.

Hugging Face's ZeroGPU images run Python 3.10, so anything imported by the
demo has to work there even though the project targets 3.11+.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover - exercised on the deployment target, not in CI
    from enum import Enum

    class StrEnum(str, Enum):
        """3.11's enum.StrEnum, including its str() behaviour.

        Plain (str, Enum) stringifies to "Class.member" before 3.11, which
        would change values already written into dataset and eval files.
        """

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
