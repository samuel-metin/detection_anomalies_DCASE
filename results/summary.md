# Dernier run : nnae/nnae-alpha01-30
- changement testé : nnae alpha 0.1, 30 epochs (10 suffisent-elles ?)
- modèle : {'name': 'nnae', 'input_shape': 640, 'hidden_shapes': [128, 128, 128, 128], 'latent_shape': 8, 'n_ids': 8, 'alpha': 0.1, 'memory_size': 128}
- données : pump, n_frames 5, batch 512 ; entraînement : {'epochs': 30, 'lr': 0.001}, 30 epochs, seed 0
- perte finale : train 12.28, val 12.16
- **AUC 75.74 / pAUC 62.68** (moyenne par ID, score = erreur moyenne)
- par ID (AUC / pAUC) : id_00 73.8/57.6, id_02 63.9/59.3, id_04 92.5/71.2, id_06 72.8/62.6
- seuil (quantile 95% de val) : 17.49 -> rappel 39.3%, fausses alertes 7.0% sur le test

Observations :
- +2.85 AUC vs baseline officielle (72.89); -0.28 vs meilleur run précédent (nnae/nnae-alpha01-10, 76.02)
- ID le plus faible : id_02 (63.9 AUC) ; perte val / train = 0.99
- conclusion : équivalent (±1 pt, bruit entre seeds)
