# Historique des runs

| run | changement testé | AUC | pAUC | conclusion |
|---|---|---|---|---|
| ae/pump-15ep | baseline du papier, 15 epochs | 72.62 | 59.62 | premier run |
| ae/pump | baseline du papier (100 epochs, référence) | 72.85 | 59.41 | équivalent (±1 pt, bruit entre seeds) |
| nnae/nnae-sans-id-10 | nnae : AE + mélange 0.5 avec le plus proche voisin normal (autre son, même machine), 10 epochs | 72.00 | 60.20 | équivalent (±1 pt, bruit entre seeds) |
| nnae/nnae-alpha025-10 | nnae alpha 0.25 (voisin pèse 75 %), 10 epochs | 74.99 | 64.28 | meilleur (> +1 pt) |
| nnae/nnae-alpha01-10 | nnae alpha 0.1 (voisin pèse 90 %), 10 epochs | 76.02 | 64.78 | meilleur (> +1 pt) |
| nnae/nnae-alpha0-10 | nnae alpha 0 (décodeur ne voit que le voisin), 10 epochs | 71.96 | 61.42 | moins bon |
| nnae/nnae-alpha01-30 | nnae alpha 0.1, 30 epochs (10 suffisent-elles ?) | 75.74 | 62.68 | équivalent (±1 pt, bruit entre seeds) |
