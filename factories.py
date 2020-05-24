import spacy

# factories = {
#     'classifier': TextClassifier.from_disk,
#     'base': spacy.load,
#     'pipeline': spacy.load,
# }

class factories:

    @classmethod
    def get(cls, item):
        return getattr(cls, item)()

    @staticmethod
    def classifier():
        from textcat import TextClassifier
        return TextClassifier.from_disk

    @staticmethod
    def base():
        return spacy.load

    @staticmethod
    def pipeline():
        return spacy.load