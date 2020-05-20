import os
import spacy
import numpy as np
from spacy.tokens import Token, Doc, Span, DocBin
from spacy.lexeme import Lexeme
from spacy.vocab import Vocab
import random
from spacy.util import minibatch, compounding 
from sklearn.model_selection import train_test_split

class BERTClassifier:
    def __init__(self, labels=None, processor=None):
        self.labels = set(labels) if labels else set()
        self.processor = processor if processor else None
        self.best_params = None
    
    def fit(self,
        X,
        y,
        val_pct=0.1,
        batch_size=16,
        learn_rate=2e-5,
        dropout=0.1,
        patience=8,
        weight_decay = 5e-3,
        L2 = 0.0,
        alpha=1e-3,
        architecture="softmax_last_hidden"):
        
        ## Preprocess Data for Training
        self.labels = set(y)
        # data = self.format_data(X, y=y)
        # train_data, val_data = train_val_split(data, val_pct)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_pct, stratify=y)
        train_data = self._format_train_data(X_train, y=y_train)

        ## Set up and validate training params
        valid_architectures = ["softmax_last_hidden", "softmax_class_vector", "softmax_tanh_class_vector", "softmax_pooler_output"]
        if architecture not in valid_architectures:
            raise ValueError(f"`architecture` must be one of {valid_architectures}")
        learn_rate = cyclic_triangular_rate(learn_rate/3, learn_rate*3, len(train_data)//batch_size*2)
        eval_window = max(100, (len(train_data) // batch_size) // 10)

        ## Set up training pipeline
        if 'trf_textcat' not in self.processor.pipe_names:
            textcat = self.processor.create_pipe(
                "trf_textcat",
                config={"architecture": architecture, "exclusive_labels": True})
            
            if len(self.labels)>1:
                for label in self.labels:
                    textcat.add_label(label)
                self.processor.add_pipe(textcat)
            else:
                raise ValueError('At least 2 unique labels must be provided, e.g.`["postive", "negative"]`')
            
        pipe_exceptions = ["trf_textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.processor.pipe_names if pipe not in pipe_exceptions]
        with self.processor.disable_pipes(*other_pipes):
            optimizer = self.processor.resume_training()
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
                        with self.processor.use_params(optimizer.averages):
                            val_acc = self.evaluate_accuracy(X_val, y_val)
                        val_tape.append((val_acc, epoch_counter, batch_counter))
                        print(f"Epoch {epoch_counter}, Batch {batch_counter} -- train_loss: {train_losses['trf_textcat']}, val_acc: {val_acc}")
                        best_score, best_epoch, best_batch = max(val_tape)
                        if val_acc == best_score:
                            self.best_params = optimizer.averages
                        elif ((batch_counter-best_batch)//eval_window) >= patience:
                            print(f"No improvement in validation set after {patience} rounds. Restoring model weights to round {best_batch}.")
                            print(f"Training Complete.")
                            train_stop = True
                            break
                        else:
                            continue
        return

    def predict(self, X, *args, **kwargs):
        if not self.processor or 'trf_textcat' not in self.processor.pipe_names:
            raise NotImplementedError(f'A trained model is required.')

        if isinstance(X, str):
            X = [X]

        if self.best_params:
            with self.processor.use_params(self.best_params):
                return [doc.cats for doc in self.processor.pipe(X)]
        else:
            return [doc.cats for doc in self.processor.pipe(X)]

    def evaluate_accuracy(self, X, y):
        '''Evaluates accuracy of model on provided data.

        Arguments:
            X List[str] -- Text to perform prediction on
            y List[str] -- Ground truth labels
        '''
        correct = 0
        for doc, actual in zip(self.processor.pipe(X),y):
            predict, _ = max(doc.cats.items(), key=lambda item: item[1])
            correct += int(predict==actual)
        return correct/len(y)

    def _format_train_data(self, X, y=None):
        if y:
            data = [(x, {"cats": {cat:(True if cat==y_ else False) for cat in self.labels}}) for x ,y_ in zip(X, y)]
        else:
            data = X
        return data
    
    @classmethod
    def load_base_model(cls, base_mdl_name):
        processor = spacy.load(base_mdl_name)
        return cls(labels=set(), processor=processor)

    def _fix_meta(self, path):
        meta_fp = os.path.join(path,'meta.json')
        meta_json = json.load(open(meta_fp,'r'))
        if 'trf_textcat' in meta_json['factories']:
            meta_json['factories']['trf_textcat'] = 'trf_textcat'
            json.dump(meta_json, open(meta_fp,'w'))

    def to_disk(self, path):
        if self.best_params:
            with self.processor.use_params(self.best_params):
                self.processor.to_disk(path)
        else:
            self.processor.to_disk(path)
        self._fix_meta(path)
        
    def from_disk(self, path):
        self.processor = spacy.load(path)
        self.meta = self.processor.get_pipe('trf_textcat').cfg
        self.labels = set(self.meta['labels'])
        self.best_params = None
    
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
        sentences = list(self.model(X).sents)
        sentence_predictions = [max(self.model(s).cats.items(), key=lambda item: item[1]) for s in sentences]
        sentence_polarities = [self.label_map[x] for x in sentence_predictions]
        sentence_polarities = [p for p in sentence_polarities if p != 0]
        if len(sentence_polarities)>0:
            return np.mean(sentence_polarities)
        else:
            return 0.0
        
    def predict(self, X, *args, **kwargs):
        if isinstance(X, str):
            return self.predict_one(X, *args, **kwargs)
        else:
            return [self.predict_one(x, *args, **kwargs) for x in X]

        

