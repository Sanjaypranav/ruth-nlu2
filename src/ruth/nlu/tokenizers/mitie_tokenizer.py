from typing import List, Text

from ruth.nlu.tokenizers.tokenizer import Token, Tokenizer
from ruth.shared.nlu.training_data.message import Message

from ruth.shared.utils.io import DEFAULT_ENCODING


class MitieTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        import mitie

        text = message.get(attribute)

        encoded_sentence = text.encode(DEFAULT_ENCODING)
        tokenized = mitie.tokenize_with_offsets(encoded_sentence)
        tokens = [
            self._token_from_offset(token, offset, encoded_sentence)
            for token, offset in tokenized
        ]

        return self._apply_token_pattern(tokens)

    def _token_from_offset(
        self, text: bytes, offset: int, encoded_sentence: bytes
    ) -> Token:
        return Token(
            text.decode(DEFAULT_ENCODING),
            self._byte_to_char_offset(encoded_sentence, offset),
        )

    @staticmethod
    def _byte_to_char_offset(text: bytes, byte_offset: int) -> int:
        return len(text[:byte_offset].decode(DEFAULT_ENCODING))
