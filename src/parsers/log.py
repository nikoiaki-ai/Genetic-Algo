from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True) 
class LogRecord: 
    participant_id: str
    task : str
    session_id: str
    trial_id: str
    completion_time: float
    dropped: Optional[bool] = None
    comments: Optional[str] = None

_RE_KV = re.compile(r"^\s*([^:]+)\s*:\s*(.+?)\s*$")
_RE_TIME = re.compile(r"^\s*Time\s+elapsed\s*:\s*([0-9]*\.?[0-9]+)\s*s?\s*$", re.IGNORECASE)
_RE_DROPPED = re.compile(r"^\s*#\s*dropped\s*:\s*(\d+)?\s*$", re.IGNORECASE)

def parse_log_file(path: str | Path) -> LogRecord: 
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    participant_id = None
    task = None
    session_id = None
    trial_id = None
    completion_time_s = None
    dropped = None
    comments = None

    for line in text:
        line = line.strip()
        
        if not line: 
            continue
        m = _RE_TIME.match(line)
        
        if m: 
            completion_time_s = float(m.group(1))
            continue
        m = _RE_DROPPED.match(line)
        
        if m: 
            dropped = bool(int(m.group(1))) if m.group(1) is not None else True
            continue
        
        m = _RE_KV.match(line)
        if m: 
            key = re.sub(r"\s+", "", m.group(1)).lower()
            value = m.group(2).strip()
            
            if key in {"participantid", "participant_id"}: 
                participant_id = value
            elif key == "task": 
                task = value
            elif key in {"sessionid", "session_id"}: 
                session_id = value
            elif key in {"trialid", "trial_id"}: 
                trial_id = value
            elif key == "comments": 
                comments = value

    missing = [k for  k,  v in {
            "ParticipantID": participant_id,
            "Task": task,
            "SessionID": session_id,
            "TrialID": trial_id,
            "Time Elapsed": completion_time_s
        }.items() if v is None]

    if missing:
        raise ValueError(f"Missing required fields in log file {path}: {', '.join(missing)}")
    return LogRecord(
        participant_id=participant_id,
        task=task,
        session_id=session_id,
        trial_id=trial_id,
        completion_time=completion_time_s,
        dropped=dropped,
        comments=comments,
    )
