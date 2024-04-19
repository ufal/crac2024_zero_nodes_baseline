#!/usr/bin/env python3

# This file is part of CRAC24 Zero Nodes Baseline
# <https://github.com/ufal/crac2024_zero_nodes_baseline>.
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import argparse
import dataclasses
import datetime
import io
import json
import os
import pickle
import re
os.environ.setdefault("KERAS_BACKEND", "torch")

import kagglehub
import keras
import numpy as np
import torch
import transformers

import crac2024_zero_nodes_eval

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--dev", default=[], nargs="+", type=str, help="Dev CoNLL-U files.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--context", default=512, type=int, help="Max context length.")
parser.add_argument("--context_right", default=50, type=int, help="Max right context length.")
parser.add_argument("--enodes_origin", default="word", choices=["word", "head"], type=str, help="Empty nodes origin.")
parser.add_argument("--enodes_per_origin", default=2, type=int, help="Max empty nodes per position.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--epochs_frozen", default=2, type=int, help="Number of epochs with frozen transformer.")
parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default="cos", choices=["none", "cos"], type=str, help="Learning rate decay.")
parser.add_argument("--learning_rate_warmup", default=5_000, type=int, help="Number of warmup steps.")
parser.add_argument("--load", default=None, type=str, help="Path to load the model from.")
parser.add_argument("--max_train_sentence_len", default=512, type=int, help="Max sentence subwords in training.")
parser.add_argument("--prediction_threshold", default=0.5, type=float, help="Prediction threshold.")
parser.add_argument("--recompute_hiddens", default=1, type=int, help="Recompute all hiddens")
parser.add_argument("--save_model", default=False, action="store_true", help="Save the model.")
parser.add_argument("--seed", default=42, type=int, help="Initial random seed.")
parser.add_argument("--steps_per_epoch", default=5_000, type=int, help="Steps per epoch.")
parser.add_argument("--task_dim", default=512, type=int, help="Task dimension size.")
parser.add_argument("--task_hidden_layer", default=2_048, type=int, help="Task hidden layer size.")
parser.add_argument("--test", default=[], nargs="+", type=str, help="Test CoNLL-U files.")
parser.add_argument("--train", default=[], nargs="+", type=str, help="Train CoNLL-U files.")
parser.add_argument("--train_sampling_exponent", default=0.5, type=float, help="Train sampling exponent.")
parser.add_argument("--transformer", default="xlm-roberta-base", type=str, help="XLM-RoBERTA model to use.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verbose", default=2, type=int, help="Verbosity")


class CorefDataset:
    @dataclasses.dataclass
    class EmptyNode:
        word_order: int
        head: int
        deprel: int  # Remapped to a numerical id already

    @dataclasses.dataclass
    class Sentence:
        conllu_lines: list[str]
        forms: list[str]
        empty_nodes: list["EmptyNode"]
        new_document: bool

    def __init__(self, path: str, args: argparse.Namespace, train_dataset: "CorefDataset" = None):
        self.path = path

        self.deprels = ["<unk>"] if train_dataset is None else train_dataset.deprels
        self.deprel_map = {"<unk>": 0} if train_dataset is None else train_dataset.deprel_map
        self.sentences = []
        self.conllu_for_eval = None if train_dataset is None else []

        # Load the CoNLL-U file
        with open(path, "r", encoding="utf-8") as file:
            in_sentence = False
            for line in file:
                if self.conllu_for_eval is not None:
                    self.conllu_for_eval.append(line)

                line = line.rstrip("\r\n")
                if not line:
                    in_sentence = False
                else:
                    if not in_sentence:
                        self.sentences.append(self.Sentence([], [], [], False))
                        in_sentence = True

                    if not re.match(r"^[0-9]*[.]", line):
                        self.sentences[-1].conllu_lines.append(line)
                        if match := re.match(r"^([0-9]+)\t([^\t]*)\t", line):
                            word_id, form = int(match.group(1)), match.group(2)
                            assert len(self.sentences[-1].forms) == word_id - 1, "Bad IDs in the CoNLL-U file"
                            self.sentences[-1].forms.append(form)
                        continue

                    columns = line.split("\t")
                    word_order = columns[0].split(".", maxsplit=1)[0]
                    head, deprel = columns[8].split("|", maxsplit=1)[0].split(":", maxsplit=1)

                    if deprel not in self.deprel_map:
                        if train_dataset is not None:
                            deprel = "<unk>"
                        else:
                            self.deprel_map[deprel] = len(self.deprels)
                            self.deprels.append(deprel)

                    self.sentences[-1].empty_nodes.append(self.EmptyNode(
                        int(word_order), int(head), self.deprel_map[deprel]))

        if self.conllu_for_eval is not None:
            self.conllu_for_eval = "".join(self.conllu_for_eval)

        # Fill new_document
        for i, sentence in enumerate(self.sentences):
            sentence.new_document = i == 0 or any(re.match(r"^\s*#\s*newdoc", line) for line in sentence.conllu_lines)

        # The dataset consists of a single treebank
        self.treebank_ranges = [(0, len(self.sentences))]

    def save_mappings(self, path: str) -> None:
        mappings = CorefDataset.__new__(CorefDataset)
        mappings.deprels = self.deprels
        with open(path, "wb") as mappings_file:
            pickle.dump(mappings, mappings_file, protocol=4)

    @staticmethod
    def from_mappings(path: str) -> "CorefDataset":
        with open(path, "rb") as mappings_file:
            mappings = pickle.load(mappings_file)
        mappings.deprel_map = {deprel: i for i, deprel in enumerate(mappings.deprels)}
        return mappings

    def write_sentence(self, output: io.TextIOBase, index: int, empty_nodes: list[EmptyNode]) -> None:
        assert index < len(self.sentences), f"Sentence index {index} out of range"

        empty_nodes_lines = {}
        for empty_node in empty_nodes:
            wo_enodes = empty_nodes_lines.setdefault(empty_node.word_order, [])
            wo_enodes.append("{}.{}\t_\t_\t_\t_\t_\t_\t_\t{}:{}\t_".format(
                empty_node.word_order, len(wo_enodes) + 1, empty_node.head, self.deprels[empty_node.deprel]))

        in_initial_comments = True
        for line in self.sentences[index].conllu_lines:
            if not line.startswith("#") and in_initial_comments:
                for empty_node in empty_nodes_lines.pop(0, []):
                    print(empty_node, file=output)
                in_initial_comments = False
            print(line, file=output)
            if match := re.match(r"^([0-9]+)\t", line):
                for empty_node in empty_nodes_lines.pop(int(match.group(1)), []):
                    print(empty_node, file=output)
        print(file=output)
        assert not empty_nodes_lines, f"Got empty nodes with incorrect word orders"


class CorefDatasetMerged(CorefDataset):
    def __init__(self, datasets: list[CorefDataset]):
        self.path = "merged"

        # Create mappings
        self.deprels = ["<unk>"]
        self.deprel_map = {"<unk>": 0}
        self.sentences = []

        # Merge sentences
        self.treebank_ranges = []
        for dataset in datasets:
            assert len(dataset.treebank_ranges) == 1
            self.treebank_ranges.append((len(self.sentences), len(self.sentences) + len(dataset.sentences)))
            for sentence in dataset.sentences:
                empty_nodes = []
                for empty_node in sentence.empty_nodes:
                    deprel = dataset.deprels[empty_node.deprel]
                    if deprel not in self.deprel_map:
                        self.deprel_map[deprel] = len(self.deprels)
                        self.deprels.append(deprel)
                    empty_nodes.append(self.EmptyNode(empty_node.word_order, empty_node.head, self.deprel_map[deprel]))
                self.sentences.append(self.Sentence(sentence.conllu_lines, sentence.forms, empty_nodes, sentence.new_document))


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, coref: CorefDataset, tokenizer: transformers.PreTrainedTokenizer, args: argparse.Namespace, training: bool):
        self.coref = coref
        self.enodes_origin = args.enodes_origin
        self.training = training

        # Tokenize the sentences
        tokenized = tokenizer([sentence.forms for sentence in coref.sentences], add_special_tokens=False, is_split_into_words=True)

        tokens, word_indices = [], []
        for i, sentence in enumerate(tokenized.input_ids):
            tokens.append(sentence)
            word_indices.append([-1])  # The future SEP token in front of the sentence
            for j in range(len(coref.sentences[i].forms)):
                span = tokenized.word_to_tokens(i, j)
                word_indices[-1].append(span.start)
            word_indices[-1] = np.array(word_indices[-1], dtype=np.int32)

        # Generate sentences in context and gold data
        trimmed_sentences = 0
        self._inputs = []
        self._outputs = []
        for i in range(len(tokens)):
            sentence = tokens[i]
            indices = word_indices[i]

            # Trim if needed
            if training and len(sentence) > args.max_train_sentence_len - 4:
                trimmed_sentences += 1
                sentence = sentence[:args.max_train_sentence_len - 4]
                while indices[-1] >= len(sentence):
                    indices = indices[:-1]

            # Generate left and right context
            left, right = [], []
            if args.context:
                left_i = i
                while left_i and not coref.sentences[left_i - 1].new_document and len(left) < args.context - 4 - len(sentence):
                    left_i -= 1
                    left = tokens[left_i] + left
                right_i = i
                while right_i + 1 < len(tokens) and not coref.sentences[right_i + 1].new_document and len(right) < args.context - 4 - len(sentence):
                    right_i += 1
                    right = right + tokens[right_i]
                right_reserve = min(args.context_right, len(right), (args.context - 4 - len(sentence)) // 2)
                left = left[len(left) - (args.context - 4 - len(sentence) - right_reserve):]
                right = right[:args.context - 4 - len(sentence) - len(left)]

            self._inputs.append([
                np.array(
                    [tokenizer.cls_token_id] + left
                    + [tokenizer.sep_token_id, tokenizer.sep_token_id] + sentence
                    + [tokenizer.sep_token_id] + right,
                    dtype=np.int32),
                indices + 1 + len(left) + 2,
            ])

            # Generate outputs in the correct format, trimming if necessary
            empty_nodes = [[] for _ in range(len(indices))]
            for empty_node in coref.sentences[i].empty_nodes:
                if args.enodes_origin == "word":
                    origin, arc = empty_node.word_order, empty_node.head
                else:
                    origin, arc = empty_node.head, empty_node.word_order
                if origin < len(indices) and arc < len(indices):
                    empty_nodes[origin].append((1, arc, empty_node.deprel))
            for i in range(len(empty_nodes)):
                empty_nodes[i].append((0, -1, -1))
                while len(empty_nodes[i]) < args.enodes_per_origin:
                    empty_nodes[i].append((-1, -1, -1))
                empty_nodes[i] = empty_nodes[i][:args.enodes_per_origin]
            empty_nodes = np.array(empty_nodes, dtype=np.int32)
            self._outputs.append([empty_nodes[..., i] for i in range(3)])

        if trimmed_sentences:
            print("Trimmed {} out of {} sentences from {}".format(trimmed_sentences, len(coref.sentences), os.path.basename(coref.path)))

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, index: int):
        inputs = [torch.from_numpy(input_) for input_ in self._inputs[index]]
        outputs = [torch.from_numpy(output) for output in self._outputs[index]]
        return inputs, outputs


class TorchDataLoader(torch.utils.data.DataLoader):
    class MergedDatasetSampler(torch.utils.data.Sampler):
        def __init__(self, coref: CorefDataset, args: argparse.Namespace):
            self._treebank_ranges = coref.treebank_ranges
            self._sentences_per_epoch = args.steps_per_epoch * args.batch_size
            self._generator = torch.Generator().manual_seed(args.seed)

            treebank_weights = np.array([r[1] - r[0] for r in self._treebank_ranges], np.float32)
            treebank_weights = treebank_weights ** args.train_sampling_exponent
            treebank_weights /= np.sum(treebank_weights)
            self._treebank_sizes = np.array(treebank_weights * self._sentences_per_epoch, np.int32)
            self._treebank_sizes[:self._sentences_per_epoch - np.sum(self._treebank_sizes)] += 1
            self._treebank_indices = [[] for _ in self._treebank_ranges]

        def __len__(self):
            return self._sentences_per_epoch

        def __iter__(self):
            indices = []
            for i in range(len(self._treebank_ranges)):
                required = self._treebank_sizes[i]
                while required:
                    if not len(self._treebank_indices[i]):
                        self._treebank_indices[i] = self._treebank_ranges[i][0] + torch.randperm(
                            self._treebank_ranges[i][1] - self._treebank_ranges[i][0], generator=self._generator)
                    indices.append(self._treebank_indices[i][:required])
                    required -= min(len(self._treebank_indices[i]), required)
            indices = torch.concatenate(indices, axis=0)
            return iter(indices[torch.randperm(len(indices), generator=self._generator)])

    def _collate_fn(self, batch):
        inputs, outputs = zip(*batch)

        batch_inputs = []
        for sequences in zip(*inputs):
            batch_inputs.append(torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1))

        batch_outputs = []
        for output in zip(*outputs):
            batch_outputs.append(torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=-1))

        return tuple(batch_inputs), tuple(batch_outputs)

    def __init__(self, dataset: TorchDataset, args: argparse.Namespace, **kwargs):
        sampler = None
        if dataset.training:
            if len(dataset.coref.treebank_ranges) == 1 and not args.steps_per_epoch:
                sampler = torch.utils.data.RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
            else:
                sampler = self.MergedDatasetSampler(dataset.coref, args)
        super().__init__(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=self._collate_fn, **kwargs)


class Model(keras.Model):
    class HFTransformerLayer(keras.layers.Layer):
        def __init__(self, transformer: str, from_pretrained: bool, **kwargs):
            super().__init__(**kwargs)
            if from_pretrained:
                self._transformer = transformers.AutoModel.from_pretrained(transformer)
            else:
                config = transformers.AutoConfig.from_pretrained(transformer)
                self._transformer = transformers.AutoModel.from_config(config)
                self._transformer.eval()

        def call(self, inputs):
            return self._transformer(keras.ops.maximum(inputs, 0), attention_mask=inputs > -1).last_hidden_state

    class GatherVectors(keras.layers.Layer):
        def call(self, params, indices):
            return keras.ops.take_along_axis(params, indices[..., np.newaxis], axis=keras.ops.ndim(indices) - 1)

    def __init__(self, coref: CorefDataset, args: argparse.Namespace):
        self._args = args

        # Create the XLM-RoBERTa backbone
        transformer = self.HFTransformerLayer(args.transformer, from_pretrained=not args.load, name="backbone")

        # Create the network
        tokens = keras.layers.Input(shape=[None], dtype="int32")
        word_indices = keras.layers.Input(shape=[None], dtype="int32")

        embeddings = transformer(tokens)
        words = self.GatherVectors()(embeddings, keras.ops.maximum(word_indices, 0))
        words = keras.layers.Dropout(args.dropout)(words)

        # Generate args.enodes_per_origin hidden states
        if args.recompute_hiddens:
            hiddens = []
            for _ in range(args.enodes_per_origin):
                hiddens.append(keras.layers.Dense(args.task_dim)(
                    keras.layers.Dropout(args.dropout)(
                        keras.layers.Dense(args.task_hidden_layer, activation="relu")(
                            keras.layers.Concatenate()([words] + hiddens)))))
        else:
            hiddens = [words]
            for _ in range(args.enodes_per_origin - 1):
                hiddens.append(keras.layers.Dense(keras.ops.shape(words)[-1])(
                    keras.layers.Dropout(args.dropout)(
                        keras.layers.Dense(args.task_hidden_layer, activation="relu")(
                            keras.layers.Concatenate()(hiddens)))))
        hiddens = keras.ops.stack(hiddens, axis=2)

        # Run the classification head
        empty_nodes = keras.layers.Dense(2)(
            keras.layers.Dropout(args.dropout)(
                keras.layers.Dense(args.task_hidden_layer, activation="relu")(
                    hiddens)))

        # Run the arc-selection head
        queries = keras.layers.Dense(args.task_dim)(
            keras.layers.Dropout(args.dropout)(
                keras.layers.Dense(args.task_hidden_layer, activation="relu")(
                    hiddens)))
        keys = keras.layers.Dense(args.task_dim)(
            keras.layers.Dropout(args.dropout)(
                keras.layers.Dense(args.task_hidden_layer, activation="relu")(
                    words)))
        queries = keras.ops.transpose(queries, axes=[0, 2, 1, 3])
        keys = keras.ops.transpose(keys[:, :, np.newaxis, :], axes=[0, 2, 3, 1])
        arc_scores = keras.ops.matmul(queries, keys) / (args.task_dim ** 0.5)
        mask = keras.ops.cast(word_indices[:, np.newaxis, np.newaxis, :] > -1, arc_scores.dtype)
        arc_scores = arc_scores * mask - 1e9 * (1 - mask)
        arc_scores = keras.ops.transpose(arc_scores, axes=[0, 2, 1, 3])

        # Run the deprel prediction head
        predicted_arcs = keras.ops.argmax(arc_scores, axis=-1)
        predicted_arc_embeddings = self.GatherVectors()(words[:, np.newaxis, :, :], predicted_arcs)
        deprels = keras.layers.Dense(len(coref.deprels))(
            keras.layers.Dropout(args.dropout)(
                keras.layers.Dense(args.task_hidden_layer, activation="relu")(
                    keras.ops.concatenate([hiddens, predicted_arc_embeddings], axis=-1))))

        super().__init__(inputs=[tokens, word_indices], outputs=[empty_nodes, arc_scores, deprels])

        if args.load:
            self.load_weights(os.path.join(args.load, "model.weights.h5"))

    def save(self, path: str, epoch : int | None = None) -> None:
        optimizer = self.optimizer
        self.optimizer = None
        self.save_weights(os.path.join(path, "model.{}weights.h5".format(f"{epoch:02d}." if epoch is not None else "")))
        self.optimizer = optimizer

    def compile(self, epoch_batches: int, frozen: bool):
        args = self._args

        if frozen:
            self.get_layer("backbone").trainable = False
            schedule = 1e-3
        else:
            self.get_layer("backbone").trainable = True
            schedule = keras.optimizers.schedules.CosineDecay(
                0. if args.learning_rate_warmup else args.learning_rate,
                args.epochs * epoch_batches - args.learning_rate_warmup,
                alpha=0.0 if args.learning_rate_decay != "none" else 1.0,
                warmup_target=args.learning_rate if args.learning_rate_warmup else None,
                warmup_steps=args.learning_rate_warmup,
            )

        super().compile(
            optimizer=keras.optimizers.Adam(schedule),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1),
        )

    def predict(self, dataloader: TorchDataLoader, save_as: str|None = None) -> str:
        coref = dataloader.dataset.coref
        enodes_origin = dataloader.dataset.enodes_origin

        conllu, sentence = io.StringIO(), 0
        for batch in dataloader:
            predictions = self.predict_on_batch(batch[0])
            for b in range(len(predictions[0])):
                sentence_len = len(coref.sentences[sentence].forms)
                is_empty_node = predictions[0][b, :sentence_len + 1, :, 1] - predictions[0][b, :sentence_len + 1, :, 0] >= \
                    np.log(self._args.prediction_threshold / (1 - self._args.prediction_threshold))
                arcs = np.argmax(predictions[1][b, :sentence_len + 1, :sentence_len + 1], axis=-1)
                deprels = np.argmax(predictions[2][b, :sentence_len + 1], axis=-1)

                empty_nodes = []
                for i in range(sentence_len + 1):
                    j = 0
                    while j < len(is_empty_node[i]) and is_empty_node[i][j]:
                        origin, arc, deprel = i, arcs[i, j], deprels[i, j]
                        if enodes_origin == "word":
                            word_order, head = origin, arc
                        else:
                            word_order, head = arc, origin
                        empty_nodes.append(coref.EmptyNode(word_order, head, deprel))
                        j += 1

                coref.write_sentence(conllu, sentence, empty_nodes)
                sentence += 1

        conllu = conllu.getvalue()
        if save_as is not None:
            if os.path.dirname(save_as):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as, "w", encoding="utf-8") as conllu_file:
                conllu_file.write(conllu)
        return conllu

    def evaluate(self, dataloader: TorchDataLoader, save_as: str|None = None) -> tuple[str, dict[str, float]]:
        conllu = self.predict(dataloader, save_as=save_as)
        evaluation = crac2024_zero_nodes_eval.evaluate(conllu, dataloader.dataset.coref.conllu_for_eval)
        if save_as is not None:
            if os.path.dirname(save_as):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as + ".eval", "w", encoding="utf-8") as eval_file:
                for metric, score in evaluation.items():
                    print("{}: f1={:.2f}%, p={:.2f}%, r={:.2f}%".format(metric, 100 * score.f1, 100 * score.p, 100 * score.r), file=eval_file)
        return conllu, evaluation


def main(params: list[str] | None = None) -> None:
    args = parser.parse_args(params)

    # If supplied, load configuration from a trained model
    if args.load:
        if not os.path.exists(args.load):
            args.load = kagglehub.model_download(args.load)
        with open(os.path.join(args.load, "options.json"), mode="r") as options_file:
            args = argparse.Namespace(**{k: v for k, v in json.load(options_file).items() if k not in [
                "dev", "exp", "load", "test", "threads", "verbose"]})
            args = parser.parse_args(params, namespace=args)
    else:
        assert args.train, "Either --load or --train must be set."

        # Create logdir
        args.logdir = os.path.join("logs", "{}{}-{}-{}-s{}".format(
            args.exp + "-" if args.exp else "",
            os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0],
            os.environ.get("SLURM_JOB_ID", ""),
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            args.seed,
        ))
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, "options.json"), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True, ensure_ascii=False, indent=2)

    # Set the random seed and the number of threads
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    if args.load:
        train = CorefDataset.from_mappings(os.path.join(args.load, "mappings.pkl"))
    else:
        train = CorefDatasetMerged([CorefDataset(path, args) for i, path in enumerate(args.train)])
        train.save_mappings(os.path.join(args.logdir, "mappings.pkl"))
    devs = [CorefDataset(path, args, train_dataset=train) for i, path in enumerate(args.dev)]
    tests = [CorefDataset(path, args, train_dataset=train) for i, path in enumerate(args.test)]

    # Create the model
    model = Model(train, args)

    # Create the datasets
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.transformer)
    if not args.load:
        train_dataloader = TorchDataLoader(TorchDataset(train, tokenizer, args, training=True), args)
    dev_dataloaders = [TorchDataLoader(TorchDataset(dataset, tokenizer, args, training=False), args) for dataset in devs]
    test_dataloaders = [TorchDataLoader(TorchDataset(dataset, tokenizer, args, training=False), args) for dataset in tests]

    # Perform prediction if requested
    if args.load:
        for dataloader in dev_dataloaders:
            model.evaluate(dataloader, save_as=os.path.splitext(
                os.path.join(args.exp, os.path.basename(dataloader.dataset.coref.path)) if args.exp else dataloader.dataset.coref.path
            )[0] + ".predicted.conllu")
        for dataloader in test_dataloaders:
            model.predict(dataloader, save_as=os.path.splitext(
                os.path.join(args.exp, os.path.basename(dataloader.dataset.coref.path)) if args.exp else dataloader.dataset.coref.path
            )[0] + ".predicted.conllu")
        return

    # Train the model
    class Evaluator(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs["learning_rate"] = keras.ops.convert_to_numpy(model.optimizer.learning_rate)
            for dataloader in dev_dataloaders + (test_dataloaders if epoch + 1 == args.epochs + args.epochs_frozen else []):
                _, metrics = model.evaluate(dataloader, save_as=os.path.splitext(
                    os.path.join(args.logdir, os.path.basename(dataloader.dataset.coref.path))
                )[0] + ".{:02d}.conllu".format(epoch + 1))
                for metric, score in metrics.items():
                    logs["{}_{}".format(os.path.splitext(os.path.basename(dataloader.dataset.coref.path))[0], metric)] = 100 * score.f1
            if args.save_model and epoch + 10 >= args.epochs + args.epochs_frozen:
                model.save(args.logdir, epoch + 1)

    evaluator = Evaluator()
    if args.epochs_frozen:
        model.compile(len(train_dataloader), frozen=True)
        model.fit(train_dataloader, epochs=args.epochs_frozen, verbose=args.verbose, callbacks=[evaluator])
    if args.epochs:
        model.compile(len(train_dataloader), frozen=False)
        model.fit(train_dataloader, initial_epoch=args.epochs_frozen, epochs=args.epochs_frozen + args.epochs, verbose=args.verbose, callbacks=[evaluator])
    if args.save_model:
        model.save(args.logdir)


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
