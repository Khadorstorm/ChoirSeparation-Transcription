# Dagstuhl ChoirSet

## Description
Dagstuhl ChoirSet (DCS) is a multitrack dataset of a cappella choral music
designed to support MIR research on choral singing. The dataset includes
recordings of an amateur vocal ensemble performing two choir pieces in full
choir and quartet settings. The audio data was recorded during an MIR seminar at
Schloss Dagstuhl using different close-up microphones to capture the individual
singers’ voices.

A detailed documentation of this dataset can be found in:

Sebastian Rosenzweig, Helena Cuesta, Christof Weiß, Frank Scherbaum,
Emilia Gómez, and Meinard Müller:
Dagstuhl ChoirSet: A Multitrack Dataset for MIR Research on Choral Singing.
Transactions of the International Society for Music Information Retrieval, X(X),
pp. 1–13, 2020. DOI: https://doi.org/10.5334/tismir.48

## Contents
audio_wav_22050_mono: Multitrack recordings with a sampling rate of 22050 Hz
annotations_csv_beat: Beat annotations
annotations_csv_F0_manual: Manually annotated F0-trajectories
annotations_csv_F0_CREPE: Automatically extracted F0-trajectories using CREPE
annotations_csv_F0_PYIN: Automatically extracted F0-trajectories using PYIN
annotations_csv_scorerepresentation: Time-Aligned score representations

## DCSToolbox
The dataset is accompanied with a Python-toolbox which can be retrieved from:
https://github.com/helenacuesta/DCStoolbox
