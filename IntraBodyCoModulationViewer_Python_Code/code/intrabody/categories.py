# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Anatomical bookkeeping: the 13 angles, the 78 pairs and their categories.

Two facts about the angle set govern most of the controls in the article and
are made explicit here.

First, the four distal angles are not independent degrees of freedom. Angle 9
("L. Distal arm") is measured on the same three keypoints as angle 1
("L. Elbow"), and likewise for the other three pairs. They are alternative
parameterizations of one measurement. The article reports the full 13-angle,
78-pair representation throughout, and uses the reduced 9-angle, 36-pair set
only as a sensitivity analysis.

Second, pairs differ in how many keypoints their defining triplets share.
Angles built from overlapping keypoints inherit correlated tracking error, so
a pair sharing keypoints can look co-modulated for reasons that have nothing
to do with coordination. The stratification by shared-keypoint count is what
separates the two, and it is the control on which the article rests its
anatomical claim.
"""
from __future__ import annotations

from itertools import combinations

# Table 1 of the article. The order is the row and column order of every
# 13 x 13 matrix in the deposit.
ANGLES = [
    ("L.Elbow", ("left_shoulder", "left_elbow", "left_wrist")),
    ("R.Elbow", ("right_shoulder", "right_elbow", "right_wrist")),
    ("L.Shoulder", ("left_hip", "left_shoulder", "left_elbow")),
    ("R.Shoulder", ("right_hip", "right_shoulder", "right_elbow")),
    ("L.Hip", ("left_shoulder", "left_hip", "left_knee")),
    ("R.Hip", ("right_shoulder", "right_hip", "right_knee")),
    ("L.Knee", ("left_hip", "left_knee", "left_ankle")),
    ("R.Knee", ("right_hip", "right_knee", "right_ankle")),
    ("L.Distal arm", ("left_elbow", "left_wrist", "left_shoulder")),
    ("R.Distal arm", ("right_elbow", "right_wrist", "right_shoulder")),
    ("L.Distal leg", ("left_knee", "left_ankle", "left_hip")),
    ("R.Distal leg", ("right_knee", "right_ankle", "right_hip")),
    ("Head/Neck", ("left_ear", "nose", "right_ear")),
]

ANGLE_NAMES = [name for name, _ in ANGLES]
N_ANGLES = len(ANGLES)

# Indices of the four geometrically dependent distal angles, and of the nine
# that remain when they are excluded for the sensitivity analysis.
DEPENDENT_INDICES = [8, 9, 10, 11]
REDUCED_INDICES = [i for i in range(N_ANGLES) if i not in DEPENDENT_INDICES]

CATEGORIES = ["Intra-limb", "Homologous", "Cross-limb", "Head-body"]

_ARM = {"Elbow", "Shoulder", "Distal arm"}
_LEG = {"Hip", "Knee", "Distal leg"}


def _joint_and_side(name):
    if name == "Head/Neck":
        return "Head", "H"
    side, joint = name.split(".", 1)
    return joint, side


def category(i, j):
    """Anatomical category of the pair formed by angles i and j.

    Intra-limb  joints within the same limb on the same body side (12 pairs)
    Homologous  the same joint on opposite sides (6 pairs)
    Cross-limb  different limbs or different sides (48 pairs)
    Head-body   the Head/Neck angle with any extremity (12 pairs)

    The four counts sum to the 78 unique pairs.
    """
    ja, sa = _joint_and_side(ANGLE_NAMES[i])
    jb, sb = _joint_and_side(ANGLE_NAMES[j])
    if "Head" in (ja, jb):
        return "Head-body"
    if ja == jb and sa != sb:
        return "Homologous"
    if sa == sb and ((ja in _ARM and jb in _ARM) or (ja in _LEG and jb in _LEG)):
        return "Intra-limb"
    return "Cross-limb"


def shared_keypoints(i, j):
    """Number of keypoints the two defining triplets have in common.

    Ranges from 0 to 3. The four pairs that share all three are the elbow with
    distal arm and knee with distal leg pairs on each side.
    """
    return len(set(ANGLES[i][1]) & set(ANGLES[j][1]))


def pair_table(indices=None):
    """All unique pairs with their category and shared-keypoint count.

    Parameters
    ----------
    indices : sequence of int, optional
        Restrict to a subset of angles. Passing ``REDUCED_INDICES`` gives the
        36 pairs of the sensitivity analysis.

    Returns
    -------
    list of dict with keys row, col, angle_row, angle_col, category,
    shared_keypoints.
    """
    if indices is None:
        indices = range(N_ANGLES)
    out = []
    for i, j in combinations(indices, 2):
        out.append({"row": i, "col": j,
                    "angle_row": ANGLE_NAMES[i], "angle_col": ANGLE_NAMES[j],
                    "category": category(i, j),
                    "shared_keypoints": shared_keypoints(i, j)})
    return out


def category_indices(indices=None):
    """Map each category to the positions of its pairs in upper-triangle order."""
    table = pair_table(indices)
    out = {c: [] for c in CATEGORIES}
    for k, row in enumerate(table):
        out[row["category"]].append(k)
    return out


# Frequency bands, in Hz. The upper edge of the micro band is the Nyquist
# limit of the 15 Hz decimated signal on which every population statistic is
# computed. That band is narrow by construction and is further attenuated by
# the 4-frame block average and the 5-frame median filter applied during
# conditioning, so its near-zero values are not evidence that fine
# co-modulation is absent.
BANDS = [("postural", 0.0, 0.5),
         ("gesture", 0.5, 5.0),
         ("micro", 5.0, 7.162)]
