# Figure Captions

## 01_dataset_composition.png
Figure 1. Number of English-Igbo sentence pairs from the Bible corpus and JHW Hugging Face corpus in the train and test splits.

## 02_test_sentence_lengths.png
Figure 2. Distribution of source and reference token lengths in the held-out Combined_test set, after applying the same tokenization used for metric calculation.

## 03_training_curves.png
Figure 3. Loss curves for the trainable models: the scratch RNN, scratch Transformer, and fine-tuned OPUS-MT model.

## 04_bleu_chrf_comparison.png
Figure 4. Corpus BLEU and chrF for all four models on the same Combined_test rows, using one shared normalization and metric implementation.

## 05_exact_match_comparison.png
Figure 5. Exact-match accuracy after shared lowercasing, punctuation tokenization, and whitespace normalization. This metric is strict for translation, so values are expected to be low.

## 06_metrics_by_source.png
Figure 6. BLEU and chrF separated by whether each held-out sentence pair came from the Bible corpus or the JHW corpus.

## 07_prediction_length_ratio.png
Figure 7. Distribution of each model's predicted length divided by the reference length. The dashed line marks equal length.

## 08_qualitative_examples_table.png
Figure 8. A compact qualitative sample showing the English input, reference Igbo sentence, and the four model outputs. The full table is saved as qualitative_translation_examples.csv and .html.
