from nlp_engine import SemanticSimilarity, OrganClassifier, TRAINING_CORPUS

classifier = OrganClassifier().train(TRAINING_CORPUS)
proba = classifier.organ_probability_vector("completely unknown words xyz123")
print([(k, round(v, 4)) for k, v in sorted(proba.items(), key=lambda item: item[1], reverse=True)])
