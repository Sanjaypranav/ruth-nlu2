from typing import Iterable, Text, Optional, Dict, List

from ruth.shared.core.domain import Domain
from ruth.shared.core.training_data.structures import StoryGraph
from ruth.shared.nlu.training_data.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    from ruth.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


async def story_graph_from_paths(
    files: List[Text],
    domain: Domain,
    template_variables: Optional[Dict] = None,
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> StoryGraph:

    from ruth.shared.core.training_data import loading

    story_steps = await loading.load_data_from_files(
        files, domain, template_variables, use_e2e, exclusion_percentage
    )
    return StoryGraph(story_steps)
