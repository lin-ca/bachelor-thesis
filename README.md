# Automatic Limerick Generation Using Recurrent Neural Networks

Bachelor's thesis project (2018) — generating structured, rhyming poetry with character-level LSTM networks.

## About

This project explores automatic generation of **limericks** using deep learning. Limericks are a constrained poetic form with a strict 5-line structure, an AABBA rhyme scheme, and anapestic meter. The challenge lies in getting a neural network to learn all of these structural rules purely from data, at the character level.

A corpus of **~90,000 limericks** was used to train stacked LSTM networks of varying depth (1–4 layers, 500 neurons each). The generated poems were then evaluated using a custom evaluation framework that objectively measures structural correctness, syllable counts, meter, and rhyme quality — including a phonetic edit distance metric for detecting imperfect rhymes.

The full thesis is included as [thesis_aline.pdf](thesis_aline.pdf).

## Architecture

- **Model:** Character-level LSTM (both stateful and stateless variants)
- **Layers:** 1–4 stacked LSTM layers with 500 neurons each
- **Vocabulary:** 33 characters (a–z, space, newline, punctuation, start/end markers)
- **Training:** 200 epochs, batch size 50, RMSprop optimizer with gradient clipping
- **Generation:** Temperature-based sampling (0.2 – 1.2) from softmax output

## Evaluation

Generated limericks are evaluated on five metrics:

| Metric | Description |
|---|---|
| **Verse count** | Does the poem have exactly 5 lines? |
| **Line length** | Are lines 3–4 shorter than lines 1, 2, 5? |
| **Syllable count** | Do syllable counts match limerick structure (8–11 / 5–7)? |
| **Meter** | Does the stress pattern follow anapestic meter? |
| **Rhyme** | Does the AABBA rhyme scheme hold? (phonetic similarity via ARPAbet) |

Rhyme evaluation uses grapheme-to-phoneme conversion and phonetic edit distance rather than simple string matching, allowing detection of both perfect and imperfect rhymes.

## Tech Stack

- Python 3.5
- Keras 2.1.5 / TensorFlow 1.6
- spaCy + CMU Pronouncing Dictionary (for phonetic analysis)
- NumPy, SciPy, Matplotlib

## Project Structure

```
neural_network/       LSTM model implementations + pre-trained models
evaluation/           Evaluation framework (syllables, meter, rhyme scoring)
preprocessing/        Data formatting and cleaning scripts
g2p_processing/       Grapheme-to-phoneme conversion pipeline
data/                 Limerick training corpus (~90k limericks)
```

## Context

This was my bachelor's thesis, completed in early 2018. At the time, RNNs/LSTMs were the go-to architecture for sequence generation tasks. The project is no longer maintained — modern LLMs have made this approach obsolete for poetry generation — but it demonstrates end-to-end ML engineering: data preprocessing, model design, training, generation, and quantitative evaluation.
