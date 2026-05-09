from valkmodel.training import ContextCurriculum


def test_context_curriculum_returns_exact_stage_lengths_and_info():
    curriculum = ContextCurriculum(stages=[8, 32, 128], steps_per_stage=10)

    assert curriculum.get_current_context_length(0) == 8
    assert curriculum.get_current_context_length(9) == 8
    assert curriculum.get_current_context_length(10) == 32
    assert curriculum.get_current_context_length(29) == 128
    assert curriculum.get_current_context_length(30) == 128
    assert curriculum.get_stage_info(11) == {"stage_index": 1, "context_length": 32, "stage_start_step": 10, "stage_end_step": 20}


def test_context_curriculum_marks_stage_boundaries_for_validation():
    curriculum = ContextCurriculum(stages=[8, 32, 128], steps_per_stage=10)

    assert curriculum.should_validate(0)
    assert not curriculum.should_validate(5)
    assert curriculum.should_validate(10)
    assert curriculum.should_validate(20)
