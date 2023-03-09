"""This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope."""

import logging
import traceback
import typing
from typing import Any, Dict, Optional, Text, Type

from ruth.nlu.classifiers.diet_classifier import DIETClassifier
from ruth.nlu.classifiers.fallback_classifier import FallbackClassifier
from ruth.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from ruth.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from ruth.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from ruth.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from ruth.nlu.extractors.duckling_entity_extractor import DucklingEntityExtractor
from ruth.nlu.extractors.entity_synonyms import EntitySynonymMapper
from ruth.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from ruth.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from ruth.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from ruth.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from ruth.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer
from ruth.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer
from ruth.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from ruth.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from ruth.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from ruth.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from ruth.nlu.model import Metadata
from ruth.nlu.selectors.response_selector import ResponseSelector
from ruth.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from ruth.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from ruth.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from ruth.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from ruth.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from ruth.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from ruth.nlu.utils.mitie_utils import MitieNLP
from ruth.nlu.utils.spacy_utils import SpacyNLP
from ruth.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from ruth.shared.exceptions import RasaException
import ruth.shared.utils.common
import ruth.shared.utils.io
import ruth.utils.io
from ruth.shared.constants import DOCS_URL_COMPONENTS

if typing.TYPE_CHECKING:
    from ruth.nlu.components import Component
    from ruth.nlu.config import RasaNLUModelConfig

logger = logging.getLogger(__name__)


# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = [
    # utils
    SpacyNLP,
    MitieNLP,
    HFTransformersNLP,
    # tokenizers
    MitieTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
    ConveRTTokenizer,
    JiebaTokenizer,
    LanguageModelTokenizer,
    # extractors
    SpacyEntityExtractor,
    MitieEntityExtractor,
    CRFEntityExtractor,
    DucklingEntityExtractor,
    EntitySynonymMapper,
    RegexEntityExtractor,
    # featurizers
    SpacyFeaturizer,
    MitieFeaturizer,
    RegexFeaturizer,
    LexicalSyntacticFeaturizer,
    CountVectorsFeaturizer,
    ConveRTFeaturizer,
    LanguageModelFeaturizer,
    # classifiers
    SklearnIntentClassifier,
    MitieIntentClassifier,
    KeywordIntentClassifier,
    DIETClassifier,
    FallbackClassifier,
    # selectors
    ResponseSelector,
]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}


class ComponentNotFoundException(ModuleNotFoundError, RasaException):
    """Raised if a module referenced by name can not be imported."""

    pass


def get_component_class(component_name: Text) -> Type["Component"]:
    """Resolve component name to a registered components class."""

    if component_name == "DucklingHTTPExtractor":
        ruth.shared.utils.io.raise_deprecation_warning(
            "The component 'DucklingHTTPExtractor' has been renamed to "
            "'DucklingEntityExtractor'. Update your pipeline to use "
            "'DucklingEntityExtractor'.",
            docs=DOCS_URL_COMPONENTS,
        )
        component_name = "DucklingEntityExtractor"

    if component_name not in registered_components:
        try:
            return ruth.shared.utils.common.class_from_module_path(component_name)

        except (ImportError, AttributeError) as e:
            # when component_name is a path to a class but that path is invalid or
            # when component_name is a class name and not part of old_style_names

            is_path = "." in component_name

            if is_path:
                module_name, _, class_name = component_name.rpartition(".")
                if isinstance(e, ImportError):
                    exception_message = f"Failed to find module '{module_name}'."
                else:
                    # when component_name is a path to a class but the path does
                    # not contain that class
                    exception_message = (
                        f"The class '{class_name}' could not be "
                        f"found in module '{module_name}'."
                    )
            else:
                exception_message = (
                    f"Cannot find class '{component_name}' in global namespace. "
                    f"Please check that there is no typo in the class "
                    f"name and that you have imported the class into the global "
                    f"namespace."
                )

            raise ComponentNotFoundException(
                f"Failed to load the component "
                f"'{component_name}'. "
                f"{exception_message} Either your "
                f"pipeline configuration contains an error "
                f"or the module you are trying to import "
                f"is broken (e.g. the module is trying "
                f"to import a package that is not "
                f"installed). {traceback.format_exc()}"
            )

    return registered_components[component_name]


def load_component_by_meta(
    component_meta: Dict[Text, Any],
    model_dir: Text,
    metadata: Metadata,
    cached_component: Optional["Component"],
    **kwargs: Any,
) -> Optional["Component"]:
    """Resolves a component and calls its load method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_meta.get("class", component_meta["name"])
    component_class = get_component_class(component_name)
    return component_class.load(
        component_meta, model_dir, metadata, cached_component, **kwargs
    )


def create_component_by_config(
    component_config: Dict[Text, Any], config: "RasaNLUModelConfig"
) -> Optional["Component"]:
    """Resolves a component and calls it's create method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_config.get("class", component_config["name"])
    component_class = get_component_class(component_name)
    return component_class.create(component_config, config)
