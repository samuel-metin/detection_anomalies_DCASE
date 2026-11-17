# Dernier run : cae/C-kapka-30
- changement testé : CAE FiLM + mauvais ID -> cible 0 (Kapka, poids 0.25), 30 ep
- modèle : {'name': 'cae', 'input_shape': 640, 'hidden_shapes': [128, 128, 128, 128], 'latent_shape': 8, 'n_ids': 8, 'conditioning': 'film', 'wrong_id_weight': 0.25, 'wrong_id_target': 0.0}
- données : pump, n_frames 5, batch 512 ; entraînement : {'epochs': 30, 'lr': 0.001}, 30 epochs, seed 0
- perte finale : train 10.56, val 2551
- **AUC 28.89 / pAUC 48.46** (moyenne par ID, score = erreur moyenne)
- par ID (AUC / pAUC) : id_00 24.7/50.8, id_02 30.0/47.8, id_04 36.6/47.4, id_06 24.3/47.9
- seuil (quantile 95% de val) : 2644 -> rappel 0.0%, fausses alertes 5.8% sur le test

Observations :
- -44.00 AUC vs baseline officielle (72.89); -46.88 vs meilleur run précédent (cae/C-concat-30, 75.77)
- ID le plus faible : id_06 (24.3 AUC) ; perte val / train = 241.63
- conclusion : moins bon
