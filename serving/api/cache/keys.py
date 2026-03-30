import hashlib
import json


def prediction_key(model_name: str, inputs: dict) -> str:
    payload = json.dumps(inputs, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"nfl:pred:{model_name}:{digest}"


def player_key(name: str) -> str:
    return f"nfl:lake:player:{name.lower()}"


def team_stats_key(abbr: str) -> str:
    return f"nfl:lake:team_stats:{abbr.lower()}"


def graph_profile_key(name: str) -> str:
    return f"nfl:lake:graph_profile:{name.lower()}"


def graph_neighbors_key(name: str, hops: int) -> str:
    return f"nfl:lake:graph_neighbors:{name.lower()}:{hops}"


def college_pipeline_key(college: str) -> str:
    return f"nfl:lake:college_pipeline:{college.lower()}"


def job_key(job_id: str) -> str:
    return f"nfl:job:{job_id}"
