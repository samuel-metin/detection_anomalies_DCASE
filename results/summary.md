# Dernier run : cae/C-concat-30-seed1
- changement testé : cae (concat), 30 epochs, seed 1
- modèle : {'name': 'cae', 'input_shape': 640, 'hidden_shapes': [128, 128, 128, 128], 'latent_shape': 8, 'n_ids': 8}
- données : pump, n_frames 5, batch 512 ; entraînement : {'epochs': 30, 'lr': 0.001}, 30 epochs, seed 1
- perte finale : train 9.875, val 10.19
- **AUC 76.39 / pAUC 65.89** (moyenne par ID, score = erreur moyenne)
- par ID (AUC / pAUC) : id_00 68.1/57.7, id_02 60.8/57.8, id_04 97.8/89.0, id_06 78.8/59.1
- seuil (quantile 95% de val) : 13.03 -> rappel 36.0%, fausses alertes 5.8% sur le test

Observations :
- +3.50 AUC vs baseline officielle (72.89); +0.62 vs meilleur run précédent (cae/C-concat-30, 75.77)
- ID le plus faible : id_02 (60.8 AUC) ; perte val / train = 1.03
- conclusion : équivalent (±1 pt, bruit entre seeds) : seed 0 confirmée
