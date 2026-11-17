# Historique des runs

| run | changement testé | AUC | pAUC | conclusion |
|---|---|---|---|---|
| ae/pump-15ep | baseline du papier, 15 epochs | 72.62 | 59.62 | premier run |
| ae/pump | baseline du papier (100 epochs, référence) | 72.85 | 59.41 | équivalent (±1 pt, bruit entre seeds) |
| cae/C-concat-30 | CAE one-hot de l'ID concaténé au code latent, 30 ep | 75.77 | 64.90 | meilleur (> +1 pt) |
| cae/C-concat-30-seed1 | cae (concat), 30 ep, seed 1 (confirmation) | 76.39 | 65.89 | équivalent (±1 pt, bruit entre seeds) |
