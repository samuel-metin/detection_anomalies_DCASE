# Dernier run : ae/pump
- changement testé : baseline du papier (100 epochs, référence)
- modèle : {'name': 'ae', 'input_shape': 640, 'hidden_shapes': [128, 128, 128, 128], 'latent_shape': 8}
- données : pump, n_frames 5, batch 512 ; entraînement : {'epochs': 100, 'lr': 0.001}, 100 epochs, seed 0
- perte finale : train 9.811, val 10.07
- **AUC 72.85 / pAUC 59.41** (moyenne par ID, score = erreur moyenne)
- par ID (AUC / pAUC) : id_00 66.1/55.4, id_02 61.1/57.3, id_04 87.3/65.6, id_06 77.0/59.3
- seuil (quantile 95% de val) : 12.96 -> rappel 27.0%, fausses alertes 6.0% sur le test

Observations :
- -0.04 AUC vs baseline officielle (72.89); +0.23 vs meilleur run précédent (ae/pump-15ep, 72.62)
- ID le plus faible : id_02 (61.1 AUC) ; perte val / train = 1.03
- conclusion : équivalent (±1 pt, bruit entre seeds)
