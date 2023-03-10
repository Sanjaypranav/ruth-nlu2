import numpy as np
import pytest

from ruth.nlu.classifiers.diet_classifier import DIETClassifier
from ruth.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from ruth.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from ruth.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from ruth.shared.nlu.training_data.training_data import TrainingData
from ruth.shared.nlu.training_data.message import Message
from ruth.nlu.featurizers.featurizer import DenseFeaturizer
from ruth.nlu.constants import FEATURIZER_CLASS_ALIAS
from ruth.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE, TEXT
from ruth.utils.tensorflow.constants import FEATURIZERS, SENTENCE, SEQUENCE, LABEL


@pytest.mark.parametrize(
    "pooling, features, expected",
    [
        (
            "mean",
            np.array([[0.5, 3, 0.4, 0.1], [0, 0, 0, 0], [0.5, 3, 0.4, 0.1]]),
            np.array([[0.5, 3, 0.4, 0.1]]),
        ),
        (
            "max",
            np.array([[1.0, 3.0, 0.0, 2.0], [4.0, 3.0, 1.0, 0.0]]),
            np.array([[4.0, 3.0, 1.0, 2.0]]),
        ),
        (
            "max",
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_calculate_cls_vector(pooling, features, expected):
    actual = DenseFeaturizer._calculate_sentence_features(features, pooling)

    assert np.all(actual == expected)


def test_flexible_nlu_pipeline():
    message = Message(data={TEXT: "This is a test message.", "intent": "test"})
    training_data = TrainingData([message, message, message, message, message])

    tokenizer = WhitespaceTokenizer()
    tokenizer.train(training_data)

    featurizer = CountVectorsFeaturizer(
        component_config={FEATURIZER_CLASS_ALIAS: "cvf_word"}
    )
    featurizer.train(training_data)

    featurizer = CountVectorsFeaturizer(
        component_config={
            FEATURIZER_CLASS_ALIAS: "cvf_char",
            "min_ngram": 1,
            "max_ngram": 3,
            "analyzer": "char_wb",
        }
    )
    featurizer.train(training_data)

    featurizer = LexicalSyntacticFeaturizer({})
    featurizer.train(training_data)

    assert len(message.features) == 6
    assert message.features[0].origin == "cvf_word"
    assert message.features[0].type == FEATURE_TYPE_SEQUENCE
    assert message.features[1].origin == "cvf_word"
    assert message.features[1].type == FEATURE_TYPE_SENTENCE
    # cvf word is also extracted for the intent
    assert message.features[2].origin == "cvf_word"
    assert message.features[2].type == FEATURE_TYPE_SEQUENCE
    assert message.features[3].origin == "cvf_char"
    assert message.features[3].type == FEATURE_TYPE_SEQUENCE
    assert message.features[4].origin == "cvf_char"
    assert message.features[4].type == FEATURE_TYPE_SENTENCE
    assert message.features[5].origin == "LexicalSyntacticFeaturizer"
    assert message.features[5].type == FEATURE_TYPE_SEQUENCE

    sequence_feature_dim = (
        message.features[0].features.shape[1] + message.features[5].features.shape[1]
    )
    sentence_feature_dim = message.features[0].features.shape[1]

    classifier = DIETClassifier(
        component_config={FEATURIZERS: ["cvf_word", "LexicalSyntacticFeaturizer"]}
    )
    model_data = classifier.preprocess_train_data(training_data)

    assert len(model_data.get(TEXT).get(SENTENCE)) == 1
    assert len(model_data.get(TEXT).get(SEQUENCE)) == 1
    assert len(model_data.get(LABEL).get(SEQUENCE)) == 1
    assert model_data.get(LABEL).get(SENTENCE) is None
    assert model_data.get(TEXT).get(SEQUENCE)[0][0].shape == (5, sequence_feature_dim)
    assert model_data.get(TEXT).get(SENTENCE)[0][0].shape == (1, sentence_feature_dim)
    assert model_data.get(LABEL).get(SEQUENCE)[0][0].shape == (1, 1)
