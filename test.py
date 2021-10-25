import unittest
import GavinTokenizers


class TokenizersTest(unittest.TestCase):
    BuiltTokenizer = None
    vocab_size = 1000
    name = "Test"

    @staticmethod
    def lines():
        with open(f"readme.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines

    @classmethod
    def setUpClass(cls) -> None:
        cls.BuiltTokenizer = GavinTokenizers.SubwordTextEncoder(cls.vocab_size, cls.name)
        cls.BuiltTokenizer.build_vocabulary(cls.lines())

    def setUp(self):

        self.SubwordTextEncoder = GavinTokenizers.SubwordTextEncoder(self.vocab_size, self.name)
        self.hello_encoded = [1073, 1102, 1109, 1109, 1112]
        self.hello_decoded = "Hello"

    def test_001_VocabEmptyOnInitialisation(self):
        self.assertEqual(self.SubwordTextEncoder.get_vocab_size(), 0)

    def test_002_NameIsInitialised(self):
        self.assertEqual(self.SubwordTextEncoder.get_name(), self.name)

    def test_003_VocabBuilds(self):
        try:
            vocab = self.lines()
            self.SubwordTextEncoder.build_vocabulary(vocab)
            self.assertEqual(self.SubwordTextEncoder.get_vocab_size(), self.vocab_size)
        except Exception as e:
            self.fail(f"Building vocabulary failed error: {e}")

    def test_004_EncodesHello(self):
        self.assertEqual(len(self.BuiltTokenizer.encode("Hello")), len(self.hello_encoded))
        self.assertEqual(self.BuiltTokenizer.encode("Hello"), self.hello_encoded)

    def test_005_DecodesHello(self):
        self.assertEqual(len(self.BuiltTokenizer.decode(self.hello_encoded)), len(self.hello_decoded))
        self.assertEqual(self.BuiltTokenizer.decode(self.hello_encoded), self.hello_decoded)
