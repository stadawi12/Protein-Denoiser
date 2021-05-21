import unittest

import numpy as np
import tensorflow as tf

from analysis import weighted_sparse_categorical_crossentropy, focal_loss,\
    custom_weighted_sparse_categorical_cross_entropy
from proteins import Sample


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        source = "test_data/mrcs/amino_4.mrc"
        self.samp = Sample("3.0", source)
        self.assertEqual(self.samp.map.shape, (32, 30, 31))

        self.cshape = 17
        self.margin = 3
        self.thresh = 0.4
        self.bg = 0.999  # needed high for synthetic maps
        self.bg_low = 0.99

    def test_decompose(self):
        # decompose dense
        self.samp.decompose(cshape=self.cshape, margin=self.margin)
        dense_rec=self.samp.no_tiles
        self.samp.tiles=None

        # decompose with high background
        self.samp.decompose(cshape=self.cshape, margin=self.margin, background_limit=self.bg, threshold=self.thresh)
        thresh_rec = self.samp.no_tiles
        self.samp.tiles = None

        # decompose with lower background (more tiles missing)
        self.samp.decompose(cshape=self.cshape, margin=self.margin, background_limit=self.bg_low, threshold=self.thresh)
        thresh_low_rec = self.samp.no_tiles

        self.assertGreater(dense_rec, thresh_rec)
        self.assertGreater(thresh_rec, thresh_low_rec)

    def test_recompose(self):
        self.samp.decompose(cshape=self.cshape, margin=self.margin)
        self.samp.recompose(tiles=True)
        self.assertTrue((self.samp.map == self.samp.map_rec).all() == True)
        self.assertEqual(self.samp.map.shape, self.samp.map_rec.shape)
        self.samp.tiles = None

        self.samp.decompose(cshape=self.cshape, margin=self.margin, background_limit=self.bg, threshold=self.thresh)
        self.samp.recompose(tiles=True)
        self.assertFalse((self.samp.map == self.samp.map_rec).all() == True)
        self.assertEqual(self.samp.map.shape, self.samp.map_rec.shape)


class TestLoss(unittest.TestCase):

    def setUp(self):
        self.target_onehot = tf.constant([1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.], shape=[5, 3])
        self.target_sparse = tf.constant([0., 1., 1., 2., 2.], shape=[5])

        # perfect prediction
        self.logits_perfect =\
            tf.constant([1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.], shape=[5, 3])
        # near perfect prediction
        self.logits_good =\
            tf.constant([.9, .05, .05, .5, .89, .6, .05, .94, .01, .05, .01, .94, .05, .05, .9], shape=[5, 3])
        # very wrong on a balanced sample
        self.logits_bad_low_penal = \
            tf.constant([.9, .05, .05, .5, .89, .6, .05, .94, .01, .05, .01, .94, .05, .9, .05], shape=[5, 3])
        # very wrong on an imbalanced sample - weighted test case
        self.logits_bad_high_penal =\
            tf.constant([.05, .05, .9, .5, .89, .6, .05, .94, .01, .05, .01, .94, .05, .05, .9], shape=[5, 3])
        # uncertain on a balanced sample - focal test case
        self.logits_bad_easier_case = \
            tf.constant([.9, .05, .05, .5, .89, .6, .05, .94, .01, .05, .01, .94, .05, .5, .45], shape=[5, 3])
        # first sample has 90% occupancy - custom weighted test case
        self.weight_matrix_high = \
            tf.constant([.9, 1., 1., 1., 1.], shape=[5])
        # last sample has 50% occupancy - custom weighted test case
        self.weight_matrix_low = \
            tf.constant([.5, 1., 1., 1., 1.], shape=[5])


    def test_categorical_vs_sparse(self):
        print("\nTEST 1: CAT VS SPARSE")

        print("\n---------------------------------------------------- CATEGORICAL")
        loss_categorical_perfect = tf.keras.backend.categorical_crossentropy(
            self.target_onehot, self.logits_perfect).numpy()
        loss_categorical_good = tf.keras.backend.categorical_crossentropy(
            self.target_onehot, self.logits_good).numpy()
        loss_categorical_bad = tf.keras.backend.categorical_crossentropy(
            self.target_onehot, self.logits_bad_high_penal).numpy()

        print("CCE zero", loss_categorical_perfect, np.sum(loss_categorical_perfect))
        print("CCE low", loss_categorical_good, np.sum(loss_categorical_good))
        print("CCE high", loss_categorical_bad, np.sum(loss_categorical_bad))

        print("\n---------------------------------------------------- SPARSE CATEGORICAL")
        loss_sparse_perfect = tf.keras.backend.sparse_categorical_crossentropy(
            self.target_sparse, self.logits_perfect).numpy()
        loss_sparse_good = tf.keras.backend.sparse_categorical_crossentropy(
            self.target_sparse, self.logits_good).numpy()
        loss_sparse_bad = tf.keras.backend.sparse_categorical_crossentropy(
            self.target_sparse, self.logits_bad_high_penal).numpy()

        print("SCCE zero", loss_sparse_perfect, np.sum(loss_sparse_perfect))
        print("SCCE low", loss_sparse_good, np.sum(loss_sparse_good))
        print("SCCE high", loss_sparse_bad, np.sum(loss_sparse_bad))

        print(np.sum(loss_categorical_perfect), np.sum(loss_sparse_perfect))
        self.assertAlmostEqual(float(np.sum(loss_categorical_perfect)), float(np.sum(loss_sparse_perfect)), places=5)
        self.assertAlmostEqual(float(np.sum(loss_categorical_good)), float(np.sum(loss_sparse_good)), places=5)
        self.assertAlmostEqual(float(np.sum(loss_categorical_bad)), float(np.sum(loss_sparse_bad)), places=5)

    def test_sparse_vs_weightedsparse(self):
        print("\nTEST 2: SPARSE VS WEIGHTED")

        print("\n---------------------------------------------------- SPARSE CATEGORICAL")
        loss_sparse_bad_lower = tf.keras.backend.sparse_categorical_crossentropy(
            self.target_sparse, self.logits_bad_low_penal).numpy()
        loss_sparse_bad_higher = tf.keras.backend.sparse_categorical_crossentropy(
            self.target_sparse, self.logits_bad_high_penal).numpy()
        print("SCCE high", loss_sparse_bad_lower, np.sum(loss_sparse_bad_lower))
        print("SCCE higher", loss_sparse_bad_higher, np.sum(loss_sparse_bad_higher))

        print("\n---------------------------------------------------- SPARSE PLAIN WEIGHTS")
        wscce = weighted_sparse_categorical_crossentropy([0, 1, 2], mode='plain_weight')
        loss_weighted_bad_lower = wscce(self.target_sparse, self.logits_bad_low_penal).numpy()
        loss_weighted_bad_higher = wscce(self.target_sparse, self.logits_bad_high_penal).numpy()
        print("WSCCE high", loss_weighted_bad_lower, np.sum(loss_weighted_bad_lower))
        print("WSCCE higher", loss_weighted_bad_higher, np.sum(loss_weighted_bad_higher))

        self.assertAlmostEqual(float(np.sum(loss_sparse_bad_higher)), float(np.sum(loss_sparse_bad_lower)), places=5)
        self.assertGreater(float(np.sum(loss_weighted_bad_higher)), float(np.sum(loss_weighted_bad_lower)))

    def test_weightedsparse_weights(self):
        print("\nTEST 3: WEIGHTED")

        print(self.target_sparse)
        print(self.logits_bad_low_penal)
        print(self.logits_bad_high_penal)

        print("\n---------------------------------------------------- SPARSE PLAIN WEIGHTS")
        wscce = weighted_sparse_categorical_crossentropy([0, 1, 2], mode='plain_weight')
        loss_weighted_bad_lower = wscce(self.target_sparse, self.logits_bad_low_penal).numpy()
        loss_weighted_bad_higher = wscce(self.target_sparse, self.logits_bad_high_penal).numpy()
        print("Custom WSCCE high", loss_weighted_bad_lower, np.sum(loss_weighted_bad_lower))
        print("Custom WSCCE higher", loss_weighted_bad_higher, np.sum(loss_weighted_bad_higher))

        print("\n---------------------------------------------------- SPARSE PENLALTY WEIGHTS")
        wscce = weighted_sparse_categorical_crossentropy([0, 1, 2], mode='penal_weight')
        loss_weighted_bad_lower_penal = wscce(self.target_sparse, self.logits_bad_low_penal).numpy()
        loss_weighted_bad_higher_penal = wscce(self.target_sparse, self.logits_bad_high_penal).numpy()
        print("WSCCE high", loss_weighted_bad_lower_penal, np.sum(loss_weighted_bad_lower_penal))
        print("WSCCE higher", loss_weighted_bad_higher_penal, np.sum(loss_weighted_bad_higher_penal))

        self.assertGreater(float(np.sum(loss_weighted_bad_higher_penal)), float(np.sum(loss_weighted_bad_higher)))
        self.assertLess(float(np.sum(loss_weighted_bad_lower_penal)), float(np.sum(loss_weighted_bad_lower)))
        self.assertGreater(loss_weighted_bad_higher_penal[0], loss_weighted_bad_higher[0])
        self.assertLess(loss_weighted_bad_lower_penal[-1], loss_weighted_bad_lower[-1])

    def test_focal_weights(self):
        print("\nTEST 4: FOCAL")

        print(self.target_sparse)
        print(self.logits_bad_low_penal)
        print(self.logits_bad_easier_case)

        print("\n---------------------------------------------------- FOCAL")
        focal = focal_loss()
        loss_focal_bad_easier = focal(self.target_sparse, self.logits_bad_easier_case).numpy()
        loss_focal_bad_lower = focal(self.target_sparse, self.logits_bad_low_penal).numpy()
        print("FOCAL high", loss_focal_bad_easier, np.sum(loss_focal_bad_easier))
        print("FOCAL higher", loss_focal_bad_lower, np.sum(loss_focal_bad_lower))

        self.assertLess(float(np.sum(loss_focal_bad_easier)), float(np.sum(loss_focal_bad_lower)))
        self.assertLess(loss_focal_bad_easier[-1], loss_focal_bad_lower[-1])

        focal = focal_loss(gamma=5)
        loss_focal_bad_high_gamma = focal(self.target_sparse, self.logits_bad_low_penal).numpy()
        focal = focal_loss(gamma=0.5)
        loss_focal_bad_low_gamma = focal(self.target_sparse, self.logits_bad_low_penal).numpy()
        print("FOCAL high gamma", loss_focal_bad_high_gamma, np.sum(loss_focal_bad_high_gamma))
        print("FOCAL low_gamma", loss_focal_bad_low_gamma, np.sum(loss_focal_bad_low_gamma))

        self.assertLess(float(np.sum(loss_focal_bad_high_gamma)), float(np.sum(loss_focal_bad_lower)))
        self.assertGreater(float(np.sum(loss_focal_bad_low_gamma)), float(np.sum(loss_focal_bad_lower)))

        focal = focal_loss(alpha=0.5)
        loss_focal_bad_high_alpha = focal(self.target_sparse, self.logits_bad_low_penal).numpy()
        focal = focal_loss(alpha=0.1)
        loss_focal_bad_low_alpha = focal(self.target_sparse, self.logits_bad_low_penal).numpy()
        print("FOCAL high alpha", loss_focal_bad_high_alpha, np.sum(loss_focal_bad_high_alpha))
        print("FOCAL low alpha", loss_focal_bad_low_alpha, np.sum(loss_focal_bad_low_alpha))

        self.assertGreater(float(np.sum(loss_focal_bad_high_alpha)), float(np.sum(loss_focal_bad_lower)))
        self.assertLess(float(np.sum(loss_focal_bad_low_alpha)), float(np.sum(loss_focal_bad_lower)))

    def test_customweightedsparse_weights(self):
        print("\nTEST 5: CUSTOM WEIGHTED")

        print(self.target_sparse)
        print(self.logits_bad_low_penal)
        print(self.weight_matrix_low)
        print(self.weight_matrix_high)

        print("\n---------------------------------------------------- SPARSE CUSTOM WEIGHTS")
        cwscce = custom_weighted_sparse_categorical_cross_entropy(self.weight_matrix_low)  # one sample downweighted
        loss_custom_weighted_bad_lower = cwscce(self.target_sparse, self.logits_bad_low_penal).numpy()
        cwscce = custom_weighted_sparse_categorical_cross_entropy(self.weight_matrix_high)
        loss_custom_weighted_bad_higher = cwscce(self.target_sparse, self.logits_bad_low_penal).numpy()
        print("Custom WSCCE high", loss_custom_weighted_bad_lower, np.sum(loss_custom_weighted_bad_lower))
        print("Custom WSCCE higher", loss_custom_weighted_bad_higher, np.sum(loss_custom_weighted_bad_higher))

        self.assertGreater(float(np.sum(loss_custom_weighted_bad_higher)), float(np.sum(loss_custom_weighted_bad_lower)))
        self.assertGreater(loss_custom_weighted_bad_higher[0], loss_custom_weighted_bad_lower[0])

    def test_customweighted_vs_sparse(self):
        print("\nTEST 6: CUSTOM WEIGHTED VS SPARSE")

        print(self.target_sparse)
        print(self.logits_bad_low_penal)
        print(self.weight_matrix_high)

        print("\n---------------------------------------------------- SPARSE CUSTOM WEIGHTS")
        cwscce = custom_weighted_sparse_categorical_cross_entropy(self.weight_matrix_low)  # one sample downweighted
        loss_custom_weighted_bad_lower = cwscce(self.target_sparse, self.logits_bad_low_penal).numpy()
        loss_sparse_bad = tf.keras.backend.sparse_categorical_crossentropy(
            self.target_sparse, self.logits_bad_low_penal).numpy()
        print("Custom WSCCE", loss_custom_weighted_bad_lower, np.sum(loss_custom_weighted_bad_lower))
        print("SCCE", loss_sparse_bad, np.sum(loss_sparse_bad))

        self.assertGreater(float(np.sum(loss_sparse_bad)), float(np.sum(loss_custom_weighted_bad_lower)))
        self.assertGreater(loss_sparse_bad[0], loss_custom_weighted_bad_lower[0])
        self.assertEqual(loss_sparse_bad[0]/2, loss_custom_weighted_bad_lower[0])

