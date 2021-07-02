from src.evaluation_metrics.segmentation_metrics import dice_score, mean_dice_score, jaccard
import numpy as np
import unittest

# segmentations for test in 2D
S1 = np.array([[0, 0, 0, 0],
               [1, 0, 3, 1],
               [1, 0, 0, 1],
               [1, 1, 1, 1]])

S2 = np.array([[0, 0, 3, 0],
               [1, 0, 3, 1],
               [1, 0, 0, 1],
               [1, 0, 0, 1]])


class TestDiceScore(unittest.TestCase):
    """
    Test class for the Dice score.
    """

    def test_2d(self):
        """
        Test the correctness of the dice score in 2D for S1 and S2.
        """
        # True values for the dice scores
        true_dice_0 = 12. / 15.
        true_dice_1 = 6. / 7.
        true_dice_2 = 1.
        true_dice_3 = 2. / 3.
        true_mean_dice = 0.25 * \
            (true_dice_0 + true_dice_1 + true_dice_2 + true_dice_3)

        # computed values for the dice score
        dice_0 = dice_score(S1, S2, fg_class=0)
        dice_1 = dice_score(S1, S2, fg_class=1)
        dice_2 = dice_score(S1, S2, fg_class=2)
        dice_3 = dice_score(S1, S2, fg_class=3)
        mean_dice = mean_dice_score(S1, S2, labels_list=[0, 1, 2, 3])

        # test correctness of the computed values
        self.assertAlmostEqual(dice_0, true_dice_0, places=4)
        self.assertAlmostEqual(dice_1, true_dice_1, places=4)
        self.assertAlmostEqual(dice_2, true_dice_2, places=4)
        self.assertAlmostEqual(dice_3, true_dice_3, places=4)
        self.assertAlmostEqual(mean_dice, true_mean_dice, places=4)


class TestJaccard(unittest.TestCase):
    """
    Test class for the Dice score.
    """

    def test_2d(self):
        """
        Test the correctness of the dice score in 2D for S1 and S2.
        """
        # True values for the dice scores
        true_jac_0 = 6. / 9.
        true_jac_1 = 3. / 4.
        true_jac_2 = 1.
        true_jac_3 = 1. / 2.

        # computed values for the dice score
        jac_0 = jaccard(S1, S2, fg_class=0)
        jac_1 = jaccard(S1, S2, fg_class=1)
        jac_2 = jaccard(S1, S2, fg_class=2)
        jac_3 = jaccard(S1, S2, fg_class=3)

        # test correctness of the computed values
        self.assertAlmostEqual(jac_0, true_jac_0, places=4)
        self.assertAlmostEqual(jac_1, true_jac_1, places=4)
        self.assertAlmostEqual(jac_2, true_jac_2, places=4)
        self.assertAlmostEqual(jac_3, true_jac_3, places=4)


if __name__ == '__main__':
    unittest.main()
