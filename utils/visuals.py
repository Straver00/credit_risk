import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def probability_to_score(prob):
    pdo = 50
    base_odds = 50
    base_score = 600
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds = (1 - prob) / prob
    score = np.maximum(offset + factor * np.log(odds), 0.0)
    return score

def user_hist_to_file(prob_user, prob_data, output_path="static/histogram.png"):
    scores = probability_to_score(prob_data)
    score_user = probability_to_score(prob_user)
    score_corte = 346.275
    bins = np.linspace(min(scores), max(scores), 500)

    scores_left = scores[scores < score_corte]
    scores_right = scores[scores >= score_corte]

    plt.figure(figsize=(10, 7))
    plt.title('Histograma del Scorecard', fontsize=18)
    plt.hist(scores_left, bins=bins, color='#FF9999', alpha=0.8, label='Susceptible a rechazo')
    plt.hist(scores_right, bins=bins, color='#A8E6A1', alpha=0.8, label='Susceptible a aprobación')

    plt.axvline(x=score_corte, color='black', linestyle='--', linewidth=2, label=f'Punto de Corte: {score_corte}')
    plt.axvline(x=score_user, color='blue', linestyle='--', linewidth=1.5, label=f'Tu puntaje: {round(score_user, 3)}')

    plt.xlabel('Scorecard', fontsize=14)
    plt.ylabel('Cantidad de Personas', fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path