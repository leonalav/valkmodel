def test_phase_4_5_and_6_utilities_are_publicly_importable():
    from valkmodel import ContextCurriculum, JEPAModule, LatentBranchingModule, PackedDataset, TrainingProfiler
    from valkmodel.training import AblationTracker, ScalingValidator

    assert JEPAModule is not None
    assert LatentBranchingModule is not None
    assert PackedDataset is not None
    assert ContextCurriculum is not None
    assert TrainingProfiler is not None
    assert ScalingValidator is not None
    assert AblationTracker is not None
