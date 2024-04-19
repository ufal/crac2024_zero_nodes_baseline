# CRAC 2024 Zero Nodes Baseline

This repository contains the source code of the CRAC 2024 Zero Nodes Baseline
system for predicting zero nodes in the input CoNLL-U files. The source is
available under the MPL-2.0 license, and the pre-trained model under the
CC BY-NC-SA license.

---

## Content of this Repository

- `crac2024_zero_nodes_baseline.py` is the source code of the whole system,
  implemented in Keras 3 with PyTorch backend.

- `crac2024_zero_nodes_eval.py` provides evaluation of predicted zero nodes,
  both as a module (used by the `crac2024_zero_nodes_baseline.py`) and also
  as a command-line tool.

## The Released `crac2024_zero_nodes_baseline` Model

The `crac2024_zero_nodes_baseline` is a `XLM-RoBERTa-large`-based multilingual
model for predicting zero nodes, trained on CorefUD 1.2 data. It is [released
on KaggleHub](https://www.kaggle.com/models/ufal-mff/crac2024_zero_nodes_baseline/)
under the CC BY-NC-SA 4.0 license, and it will also be downloaded
automatically by `crac2024_zero_nodes_baseline` when running inference using the
`--load ufal-mff/crac2024_zero_nodes_baseline/keras/1` argument.

The model was used to generate baseline zero nodes prediction in the
[CRAC 2024 Shared Task on Multilingual Coreference Resolution](https://ufal.mff.cuni.cz/corefud/crac24).

The model is language agnostic, so in theory it can be used to
predict coreference in any `XLM-RoBERTa` language.

## Training a Single Multilingual `XLM-RoBERTa-large`-based Model

To train a single multilingual model on all the data using `XLM-RoBERTa-large`, you should
1. download the CorefUD 1.2 data,
2. create a Python environments with the packages listed in `requirements.txt`,
3. train the model itself using the `crac2024_zero_nodes_baseline.py` script.

   The released model has been trained using the following command:
   ```sh
   tbs="ca-ancora cs-pcedt cs-pdt cu-proiel es-ancora grc-proiel hu-korkor hu-szeged pl-polishcoreferencecorpus tr-itcc"
   python3 crac24_zero_nodes_baseline.py $(for mode in train dev test; do echo --$mode; for tb in $tbs; do echo data/$tb-$mode.conllu; done; done) --exp xlmr-large --batch_size=64 --context=0 --max_train_sentence_len=120 --learning_rate=1e-5 --transformer=xlm-roberta-large --enodes_origin=head --seed=3 --save_model
   ```
   It assumes the training files are available in `data/{treebank}-{train/dev/test}.conllu`,
   with `train` and `dev` files containing the gold empty nodes and `test` data
   without empty nodes (i.e., they are not evaluated).

## Predicting with a Trained Model.

To predict with the released `crac2024_zero_nodes_baseline` model, use the following arguments:
```sh
crac2024_zero_nodes_baseline.py --load ufal-mff/crac2024_zero_nodes_baseline/keras/1 --exp target_directory --test input1.conllu input2.conllu
```
- instead of a KaggleHub identifier, you can use directory name – if the given
  path name exists, model is loaded from it;
- the outputs are generated in the target directory, with `.predicted.conllu` suffix;
- if you want to also evaluate the predicted files, you can use `--dev` option instead of `--test`;
  that way, another file with `.predicted.conllu.eval` suffix will be created by `crac2024_zero_nodes_eval.py`.

## Evaluation of Zero Nodes Prediction Performance

The `crac2024_zero_nodes_eval.py` performs intrinsic evaluation of zero nodes
prediction. It computes F1-score, precision, and recall in several settings:
- `WO`: a predicted zero node is considered correct if it has correct word
  order (i.e., the value of the CoNLL-U first column before a dot);
- `ARC`: a predicted zero node is considered correct if it has correct
  parent in the `DEPS` column (but not necessarily a correct DEPREL);
- `DEP`: a predicted zero node is considered correct if it has correct
  `DEPS` column (so both the parent and dependency relation);
- `WO_DEP`: a predicted zero node is considered correct if all
  of its word order, parent, and dependency relation is correct.

### Evaluation of the Released `crac2024_zero_nodes_baseline` Model

The following table contains the F1-scores of the released
`crac24_zero_nodes_baseline` model on the CorefUD 1.2 development data.

| Treebank                   |   WO   |  ARC   |  DEP   | WO_DEP |
|:---------------------------|-------:|-------:|-------:|-------:|
| ca-ancora                  | 93.08% | 93.73% | 93.73% | 91.66% |
| cs-pcedt                   | 70.29% | 70.08% | 67.92% | 67.81% |
| cs-pdt                     | 78.07% | 78.20% | 76.37% | 76.19% |
| cu-proiel                  | 80.82% | 81.08% | 80.55% | 80.16% |
| es-ancora                  | 92.50% | 95.06% | 95.06% | 91.98% |
| grc-proiel                 | 89.79% | 89.51% | 88.39% | 88.39% |
| hu-korkor                  | 69.57% | 74.20% | 74.20% | 66.67% |
| hu-szeged                  | 90.71% | 91.15% | 91.15% | 90.71% |
| pl-polishcoreferencecorpus | 89.61% | 89.72% | 89.61% | 89.51% |
| tr-itcc                    | 85.80% | 85.80% | 85.80% | 85.80% |
