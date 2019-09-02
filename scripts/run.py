from magpie import Magpie

magpie = Magpie()
magpie.init_word_vectors('data/hep-categories', vec_dim=100)
labels = ["Astrophysics", "Experiment-HEP", "Gravitation and Cosmology", "Phenomenology-HEP", "Theory-HEP",]
magpie.train('data/hep-categories', labels, test_ratio=0.2, epochs=30)
print(magpie.predict_from_text('Stephen Hawking studies black holes'))
