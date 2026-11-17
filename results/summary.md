# Dernier run : cae/C-film-30-seed1
- changement testé : CAE film, 30 ep, seed 1 (confirmation)
- modèle : {'name': 'cae', 'input_shape': 640, 'hidden_shapes': [128, 128, 128, 128], 'latent_shape': 8, 'n_ids': 8, 'conditioning': 'film', 'wrong_id_weight': 0.0, 'wrong_id_target': 0.0}
- données : pump, n_frames 5, batch 512 ; entraînement : {'epochs': 30, 'lr': 0.001}, 30 epochs, seed 1
- perte finale : train 9.913, val 10.16
- **AUC 73.74 / pAUC 60.98** (moyenne par ID, score = erreur moyenne)
- par ID (AUC / pAUC) : id_00 67.0/55.9, id_02 61.4/57.8, id_04 91.0/70.7, id_06 75.6/59.4
- seuil (quantile 95% de val) : 12.85 -> rappel 30.9%, fausses alertes 6.5% sur le test

Observations :
- +0.85 AUC vs baseline officielle (72.89); -2.65 vs meilleur run précédent (cae/C-concat-30-seed1, 76.39)
- ID le plus faible : id_02 (61.4 AUC) ; perte val / train = 1.03
- conclusion : moins bon
