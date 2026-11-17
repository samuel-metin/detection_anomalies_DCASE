# Historique des runs

| run | changement testé | AUC | pAUC | conclusion |
|---|---|---|---|---|
| ae/pump-15ep | baseline du papier, 15 epochs | 72.62 | 59.62 | premier run |
| ae/pump | baseline du papier (100 epochs, référence) | 72.85 | 59.41 | équivalent (±1 pt, bruit entre seeds) |
| idnn/B-papier30 | IDNN papier 64-32-16, 30 ep | 72.63 | 59.85 | équivalent (±1 pt, bruit entre seeds) |
| ae_plus/A-norm-30ep | ae_plus : entrées normalisées par dimension, erreur dans l'espace normalisé (30 ep) | 73.16 | 60.63 | équivalent (±1 pt, bruit entre seeds) |
| cae/C-film-30 | CAE FiLM sur le code latent, 30 epochs | 75.16 | 63.20 | meilleur (> +1 pt) |
| idnn/B-128x3-16-30 | IDNN 128x3, goulot 16, 30 ep | 74.30 | 61.05 | équivalent (±1 pt, bruit entre seeds) |
| ae_plus/A-dae-cos-30ep | ae_plus normalisé + débruitage (bruit 0.2) + init Keras + cosine (30 ep) | 73.11 | 60.48 | moins bon |
| cae/C-concat-30 | CAE one-hot de l'ID concaténé au code latent, 30 ep | 75.77 | 64.90 | équivalent (±1 pt, bruit entre seeds) |
| idnn/B-128x3-16-f7-30 | IDNN 128x3, goulot 16, 7 trames, 30 ep | 74.39 | 60.99 | moins bon |
| ae_plus/A-256-16-30ep | ae_plus normalisé, couches 256x4, goulot 16 (30 ep) | 73.36 | 61.02 | moins bon |
| cae/C-kapka-30 | CAE FiLM + mauvais ID -> cible 0 (Kapka, poids 0.25), 30 ep | 28.89 | 48.46 | moins bon |
