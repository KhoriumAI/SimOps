from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class GeometryStatus(Enum):
    PRISTINE = "PRISTINE"       # File was perfect, no changes
    RESTORED = "RESTORED"       # File was broken, now fixed
    TERMINAL = "TERMINAL"       # File is unrecoverable
    WARNING = "WARNING"         # File has issues but might mesh (Fail Open)

@dataclass
class GuardianResult:
    status: GeometryStatus
    output_path: str
    report_path: str
    original_path: str
    lifecycle: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def was_modified(self) -> bool:
        return self.status == GeometryStatus.RESTORED