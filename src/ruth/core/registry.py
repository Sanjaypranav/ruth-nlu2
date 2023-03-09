"""This module imports all of the components. To avoid cycles, no component
should import this in module scope."""

import logging
from typing import Text, Type, TYPE_CHECKING

import ruth.shared.utils.common

if TYPE_CHECKING:
    from ruth.core.policies.policy import Policy
    from ruth.core.featurizers.tracker_featurizers import TrackerFeaturizer
    from ruth.core.featurizers.single_state_featurizer import SingleStateFeaturizer

logger = logging.getLogger(__name__)


def policy_from_module_path(module_path: Text) -> Type["Policy"]:
    """Given the name of a policy module tries to retrieve the policy.

    Args:
        module_path: a path to a policy

    Returns:
        a :class:`ruth.core.policies.policy.Policy`
    """
    try:
        return ruth.shared.utils.common.class_from_module_path(
            module_path, lookup_path="ruth.core.policies.registry"
        )
    except ImportError:
        raise ImportError(f"Cannot retrieve policy from path '{module_path}'")


def featurizer_from_module_path(module_path: Text) -> Type["TrackerFeaturizer"]:
    """Given the name of a featurizer module tries to retrieve it.

    Args:
        module_path: a path to a featurizer

    Returns:
        a :class:`ruth.core.featurizers.tracker_featurizers.TrackerFeaturizer`
    """
    try:
        return ruth.shared.utils.common.class_from_module_path(
            module_path, lookup_path="ruth.core.featurizers.tracker_featurizers"
        )
    except ImportError:
        raise ImportError(f"Cannot retrieve featurizer from path '{module_path}'")


def state_featurizer_from_module_path(
    module_path: Text,
) -> Type["SingleStateFeaturizer"]:
    """Given the name of a single state featurizer module tries to retrieve it.

    Args:
        module_path: a path to a single state featurizer

    Returns:
        a :class:`ruth.core.featurizers.single_state_featurizer.SingleStateFeaturizer`
    """
    try:
        return ruth.shared.utils.common.class_from_module_path(
            module_path, lookup_path="ruth.core.featurizers.single_state_featurizer"
        )
    except ImportError:
        raise ImportError(f"Cannot retrieve featurizer from path '{module_path}'")
