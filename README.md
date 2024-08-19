# F20AA-2023-2024 Kaggle Competition

## Overview

This repository contains the code for the F20AA-2024 [Kaggle Competition](https://www.kaggle.com/competitions/f20f21-aa-2023-2024-cw2/leaderboard). The project involves an in-depth analysis and building machine learning models to compete in a Kaggle challenge corresponding to the Text Analytics course at Heriot-Watt. The dataset contained approx 370k reviews left on amazon products of the "arts and crafts" genre. My team achieved first place in a kaggle competition through the usage of Bert-Large, a no-evaluation strategy, and a last minute gamble.

## Repository Structure

- **BERT**: Contains standalone scripts related to the BERT and BERT-LARGE models.
- **BERT/output_logs**: Contains the output logs from the standalone scripts.
- **Notebooks**: Contains Jupyter notebooks used for data exploration, experimentation, model building, and evaluation.
- **data**: Contains csv files of the original datasets and their derivatives.
- **results**: Contains csv files of the results from model evaluations. *note that filenames ending with "Changed" are simply using the labels 1-5 instead of 0-4 which is the raw output of our models, this change was made to comply with the requirements of the kaggle submission system*.
