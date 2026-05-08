__all__ = ["MultipleChoiceScoringService"]


def __getattr__(name: str):
    if name == "MultipleChoiceScoringService":
        from domain.scoring.scoring_service import MultipleChoiceScoringService

        return MultipleChoiceScoringService
    raise AttributeError(name)
