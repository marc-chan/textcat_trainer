import os
import json
import spacy
import numpy as np
from spacy.tokens import Token, Doc, Span, DocBin
from spacy.lexeme import Lexeme
from spacy.vocab import Vocab
import random
import json
from spacy.util import minibatch, compounding 
from sklearn.model_selection import train_test_split
from data import BASE_MODELS, find, load


class TextClassifier:
    pipe_name = 'textcat'
    valid_architectures = ['ensemble', 'simple_cnn', 'bow']

    def __init__(self, base=None, labels=None, processor=None):
        if not processor:
            if base:
                try:
                    processor = load(base, typ='base')
                except ValueError:
                    processor = None
            if not processor:
                raise ValueError(f"Specify a valid base model to train from. Available: {BASE_MODELS}")

        self.labels = set(labels) if labels else set()
        self.processor = processor if processor else None
        self.best_params = None

        if self.processor.meta.get('lang_factory','') == 'trf':
            self.__class__ = DeepTextClassifier

    def _setup_pipeline(self, architecture, **kwargs):
        if architecture not in self.valid_architectures:
            raise ValueError(f"`architecture` must be one of {self.valid_architectures}")
        if self.pipe_name not in self.processor.pipe_names:
            component = self.processor.create_pipe(
                self.pipe_name,
                config={"architecture": architecture, "exclusive_classes": True, **kwargs})
            
            if len(self.labels)>1:
                for label in self.labels:
                    component.add_label(label)
                self.processor.add_pipe(component)
            else:
                raise ValueError('At least 2 unique labels must be present, e.g.`["postive", "negative"]`')
        return
    
    def fit(self,
        X,
        y,
        val_pct=0.05,
        batch_size=8,
        learn_rate=2e-5,
        dropout=0.1,
        patience=5,
        L2 = 0.0,
        architecture="simple_cnn",
        ngram_size=2,):
        
        ## Preprocess Data for Training
        self.labels = set(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_pct, stratify=y)
        train_data = self._format_train_data(X_train, y=y_train)

        ## Set up training pipeline
        self._setup_pipeline(architecture, ngram_size=ngram_size, attr="lower")
            
        pipe_exceptions = ["textcat"]
        other_pipes = [pipe for pipe in self.processor.pipe_names if pipe not in pipe_exceptions]
        with self.processor.disable_pipes(*other_pipes):
            optimizer = self.processor.begin_training()
            learn_rate = cyclic_triangular_rate(learn_rate/3, learn_rate*3, len(train_data)//batch_size*2)
            eval_window = max(50, (len(train_data) // batch_size) // 10)
            optimizer.L2 = L2
            val_tape = []
            epoch_counter = 0
            batch_counter = 0
            train_stop = False
            print(f"Training begins, evaluating every {eval_window} batches of {batch_size} training examples")
            while not train_stop:
                random.shuffle(train_data)
                epoch_counter += 1
                for batch in minibatch(train_data, size=batch_size):
                    train_losses = {}
                    batch_counter += 1
                    texts, cats = zip(*batch)
                    optimizer.alpha = next(learn_rate)
                    self.processor.update(texts, cats, sgd=optimizer, drop=dropout, losses=train_losses)
                    if (batch_counter % eval_window) == 0:
                        val_acc = self.evaluate_accuracy(X_val, y_val, params=optimizer.averages)
                        val_tape.append((val_acc, epoch_counter, batch_counter))
                        print(f"Eval Round {batch_counter//eval_window} (epoch {epoch_counter}) -- train_loss: {train_losses[self.pipe_name]}, val_acc: {val_acc}")
                        best_score, best_epoch, best_batch = max(val_tape)
                        if val_acc == best_score:
                            self.best_params = optimizer.averages
                        elif ((batch_counter-best_batch)//eval_window) >= patience:
                            print(f"No improvement in validation accuracy after {patience} rounds. Restoring model weights to evaluation round {best_batch//eval_window}.")
                            print(f"Training Complete.")
                            train_stop = True
                            break
        return
        
    def predict(self, X, params=None, *args, **kwargs):
        if not self.processor:
            raise NotImplementedError(f'A trained model is required.')

        if isinstance(X, str):
            X = [X]

        params = params if params else self.best_params
        if params:
            with self.processor.use_params(params):
                return [doc.cats for doc in self.processor.pipe(X)]
        else:
            return [doc.cats for doc in self.processor.pipe(X)]

    def evaluate_accuracy(self, X, y, params=None):
        '''Evaluates accuracy of model on provided data.

        Arguments:
            X List[str] -- Text to perform prediction on
            y List[str] -- Ground truth labels
        '''
        correct = 0
        for prediction, actual in zip(self.predict(X, params=params),y):
            predict, _ = max(prediction.items(), key=lambda item: item[1])
            correct += int(predict==actual)
        return correct/len(y)

    def _format_train_data(self, X, y=None):
        if y:
            data = [(x, {"cats": {cat:(True if cat==y_ else False) for cat in self.labels}}) for x ,y_ in zip(X, y)]
        else:
            data = X
        return data

    def _fix_meta(self, path):
        meta_fp = os.path.join(path,'meta.json')
        meta_json = json.load(open(meta_fp,'r'))
        if 'trf_textcat' in meta_json['factories']:
            meta_json['factories']['trf_textcat'] = 'trf_textcat'
        meta_json['type'] = 'classifier'
        json.dump(meta_json, open(meta_fp,'w'))

    def to_disk(self, path):
        if self.best_params:
            with self.processor.use_params(self.best_params):
                self.processor.to_disk(path)
        else:
            self.processor.to_disk(path)
        self._fix_meta(path)
        
    @classmethod
    def from_disk(cls, path):
        processor = spacy.load(path)
        pipename = 'trf_textcat' if 'trf_textcat' in processor.pipe_names else 'textcat'
        meta = processor.get_pipe(pipename).cfg
        labels = set(meta['labels'])
        clf = cls(labels=labels, processor=processor)
        clf.meta = meta
        return clf

class DeepTextClassifier(TextClassifier):
    pipe_name = 'trf_textcat'
    valid_architectures = ["softmax_last_hidden", "softmax_class_vector", "softmax_tanh_class_vector", "softmax_pooler_output"]

    def __init__(self, labels=None, processor=None):
        self.labels = set(labels) if labels else set()
        self.processor = processor if processor else None
        self.best_params = None
    
    def fit(self,
        X,
        y,
        val_pct=0.05,
        batch_size=8,
        learn_rate=2e-5,
        dropout=0.1,
        patience=5,
        weight_decay = 5e-3,
        L2 = 0.0,
        alpha=1e-3,
        architecture="softmax_last_hidden"):
        
        ## Preprocess Data for Training
        self.labels = set(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_pct, stratify=y)
        train_data = self._format_train_data(X_train, y=y_train)

        ## Set up training pipeline
        self._setup_pipeline(architecture)
            
        pipe_exceptions = ["trf_textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.processor.pipe_names if pipe not in pipe_exceptions]
        with self.processor.disable_pipes(*other_pipes):
            optimizer = self.processor.resume_training()
            learn_rate = cyclic_triangular_rate(learn_rate/3, learn_rate*3, len(train_data)//batch_size*2)
            eval_window = max(50, (len(train_data) // batch_size) // 10)
            optimizer.alpha = alpha
            optimizer.trf_weight_decay = weight_decay
            optimizer.L2 = L2
            val_tape = []
            epoch_counter = 0
            batch_counter = 0
            train_stop = False
            print(f"Training begins, evaluating every {eval_window} batches of {batch_size} training examples")
            while not train_stop:
                random.shuffle(train_data)
                epoch_counter += 1
                for batch in minibatch(train_data, size=batch_size):
                    train_losses = {}
                    batch_counter += 1
                    texts, cats = zip(*batch)
                    optimizer.trf_lr = next(learn_rate)
                    self.processor.update(texts, cats, sgd=optimizer, drop=dropout, losses=train_losses)
                    if (batch_counter % eval_window) == 0:
                        val_acc = self.evaluate_accuracy(X_val, y_val, params=optimizer.averages)
                        val_tape.append((val_acc, epoch_counter, batch_counter))
                        print(f"Eval Round {batch_counter//eval_window} (epoch {epoch_counter}) -- train_loss: {train_losses[self.pipe_name]}, val_acc: {val_acc}")
                        best_score, best_epoch, best_batch = max(val_tape)
                        if val_acc == best_score:
                            self.best_params = optimizer.averages
                        elif ((batch_counter-best_batch)//eval_window) >= patience:
                            print(f"No improvement in validation accuracy after {patience} rounds. Restoring model weights to evaluation round {best_batch//eval_window}.")
                            print(f"Training Complete.")
                            train_stop = True
                            break
        return

    
def cyclic_triangular_rate(min_lr, max_lr, period):
    it = 1
    while True:
        # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
        cycle = np.floor(1 + it / (2 * period))
        x = np.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield min_lr + (max_lr - min_lr) * relative
        it += 1

class SentimentClassifier:
    def __init__(self, label_map=None, model=None):
        self.label_map = label_map if label_map else {'positive':1, 'negative':-1, 'neutral':0}
        self.model = model

    def predict_one(self, X, *args, **kwargs):
        to_disable = [p for p in self.model.pipe_names if p not in {'parser','sentencizer'}]
        with self.model.disable_pipes(*to_disable):
            sentences = [s.text for s in self.model(X).sents]
        sentence_predictions = [max(d.cats.items(), key=lambda item: item[1]) for d in self.model.pipe(sentences)]
        sentence_polarities = [self.label_map[label] for label, score in sentence_predictions]
        sentence_polarities = [p for p in sentence_polarities if p != 0]
        if len(sentence_polarities)>0:
            return np.mean(sentence_polarities)
        else:
            return 0.0
        
    def predict(self, X, *args, **kwargs):
        if isinstance(X, Doc) or isinstance(X, Span):
            X = X.text
        if isinstance(X, str):
            return self.predict_one(X, *args, **kwargs)
        else:
            return [self.predict(x, *args, **kwargs) for x in X]

