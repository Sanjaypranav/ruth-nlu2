import logging
from typing import Dict, List, Optional, Text, Union

import ruth.shared.data
from ruth.shared.core.training_data.structures import StoryGraph
from ruth.shared.importers import utils
from ruth.shared.importers import autoconfig
from ruth.shared.importers.importer import TrainingDataImporter
from ruth.shared.importers.autoconfig import TrainingType
from ruth.shared.nlu.training_data.training_data import TrainingData
from ruth.shared.core.domain import InvalidDomain, Domain
import ruth.shared.utils.io

logger = logging.getLogger(__name__)


class RasaFileImporter(TrainingDataImporter):
    """Default `TrainingFileImporter` implementation."""

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ):

        self._domain_path = domain_path

        self._nlu_files = ruth.shared.data.get_data_files(
            training_data_paths, ruth.shared.data.is_nlu_file
        )
        self._story_files = ruth.shared.data.get_data_files(
            training_data_paths, ruth.shared.data.is_story_file
        )
        self._conversation_test_files = ruth.shared.data.get_data_files(
            training_data_paths, ruth.shared.data.is_test_stories_file
        )

        self.config = autoconfig.get_configuration(config_file, training_type)

    async def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self.config

    async def get_stories(
        self,
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return await utils.story_graph_from_paths(
            self._story_files,
            await self.get_domain(),
            template_variables,
            use_e2e,
            exclusion_percentage,
        )

    async def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return await utils.story_graph_from_paths(
            self._conversation_test_files, await self.get_domain(), use_e2e=True,
        )

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        return utils.training_data_from_paths(self._nlu_files, language)

    async def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        domain = Domain.empty()

        # If domain path is None, return an empty domain
        if not self._domain_path:
            return domain
        try:
            domain = Domain.load(self._domain_path)
        except InvalidDomain as e:
            ruth.shared.utils.io.raise_warning(
                f"Loading domain from '{self._domain_path}' failed. Using "
                f"empty domain. Error: '{e}'"
            )

        return domain
