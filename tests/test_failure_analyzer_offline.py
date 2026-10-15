"""Offline pipeline-shape test for the H2 failure analyzer."""

import json

from evolvevla.diagnosis.failure_analyzer import (
    FAILURE_ANALYSIS_PROMPT,
    FAILURE_MODES,
    FailureAnalyzer,
)


def test_failed_trajectory_prompt_and_mock_response() -> None:
    analyzer = FailureAnalyzer(
        provider_model="offline-test-model",
        provider_key_id="offline-test-key-id",
    )
    trajectory = {
        "task_name": "CloseFridge",
        "episode_idx": 0,
        "seed": 42,
        "success": False,
        "total_reward": 0.0,
        "num_steps": 4,
        "steps": [
            {
                "step": 0,
                "reward": 0.0,
                "done": False,
                "ee_position": [0.0, 0.0, 0.4],
                "gripper_state": "open",
                "info": {},
            },
            {
                "step": 1,
                "reward": 0.0,
                "done": False,
                "ee_position": [0.1, 0.0, 0.4],
                "gripper_state": "closed",
                "info": {},
            },
            {
                "step": 2,
                "reward": 0.0,
                "done": False,
                "ee_position": [0.2, 0.0, 0.4],
                "gripper_state": "closed",
                "info": {"collision": True},
            },
            {
                "step": 3,
                "reward": 0.0,
                "done": True,
                "ee_position": [0.15, 0.0, 0.4],
                "gripper_state": "open",
                "info": {},
            },
        ],
        "metadata": {},
    }

    features = analyzer._extract_trajectory_features(trajectory)
    assert features["task_description"] == "CloseFridge"
    assert features["num_steps"] == 4
    assert features["events"]

    prompt = analyzer._format_analysis_prompt(features)
    assert FAILURE_ANALYSIS_PROMPT
    assert "CloseFridge" in prompt
    assert "Classify this failure into exactly ONE primary category" in prompt
    for mode in FAILURE_MODES:
        assert mode in prompt
    assert '"failure_mode": "<category_name>"' in prompt

    mock_response = {
        "failure_mode": "collision",
        "confidence": 0.91,
        "failure_step": 2,
        "description": "The end effector contacts the environment during approach.",
        "root_cause": "The approach path lacks sufficient clearance.",
        "suggested_data": "Add tight-clearance approach demonstrations with collision labels.",
    }
    parsed = analyzer._parse_and_validate_response(json.dumps(mock_response))
    assert parsed == mock_response


if __name__ == "__main__":
    test_failed_trajectory_prompt_and_mock_response()
    print("offline failure-analyzer pipeline shape: OK")
