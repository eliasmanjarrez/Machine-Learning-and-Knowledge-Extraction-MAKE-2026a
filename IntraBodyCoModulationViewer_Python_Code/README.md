# IntraBodyCoModulationViewer
Intra-body joint-angle co-modulation: analysis pipeline and derived data

Companion repository for

> Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E.
> **Interactive playback visualizer to analyse joint-angle co-modulation with a
> wavelet approach: application to pose-voice relationships during spontaneous
> conversation.**

Repository: [repository URL to be inserted]
Article: [article DOI to be inserted]

**If you use this code or these data in your own work, please cite the
article.** A machine-readable citation is in `CITATION.cff`. The article is the
appropriate reference for the method, the dataset and the visualizer alike.

---

## The one thing to understand before anything else

**Nothing here reads video.** The pipeline and the visualizer both start from a
table of body coordinates, and producing that table is a separate step that you
run yourself, with DeepLabCut.

```
   your video (.mp4)
        │
        │   DeepLabCut + SuperAnimal-HumanBody       ← you run this, once
        ▼
   coordinates (.h5 and .csv)     17 keypoints per person, per frame
        │
        │   this repository
        ▼
   13 joint signals  →  wavelet analysis  →  figures, tables, exported data
```

Point the visualizer at an `.mp4` and nothing will happen: it is a web page,
and it has no pose-estimation model inside it. Section 3 says which model to
use, where its documentation lives, how the analysis in the article was run,
and how long it takes.

---

## 1. What is here, and what it is for

The repository lets a reader recompute every quantity the article reports, from
the conditioned joint-angle series through to the last p-value, without access
to the source video.

```
README.md            this file
LICENSE              MIT for the code, CC-BY-4.0 for the data
CITATION.cff         machine-readable citation
requirements.txt     what the analysis needs (three packages)
requirements-deeplabcut.txt   only if you will analyse your own video

code/                the analysis pipeline
  intrabody/         the library: wavelet core, anatomy, statistics
  step1 ... step6    one script per analysis, each printing its own numbers
  reproduce_all.py   runs all six and reports what matched

data/                de-identified derived data, 40 MB in total
viewer/              the interactive playback visualizer and three examples
tools/               dlc_h5_to_csv.py, to feed DeepLabCut output to the viewer
```

The quickest way to see whether the code works:

```bash
cd code
python reproduce_all.py
```

That takes about five minutes. Every step prints its recomputed values beside
the values printed in the article, and the final line counts how many matched.
`python reproduce_all.py --quick` runs the same checks with a reduced first
step, in under a minute.

Requirements: Python 3.9 or later with `numpy`, `scipy` and `pandas`. Nothing
else. No GPU, no network access, no installation step.

---

## 2. The study in one paragraph

Seventy adults were recorded on a single consumer camera while holding a
spontaneous conversation. A pretrained pose-estimation network located 17
anatomical keypoints per frame, from which 13 joint angles were derived. Every
pair of angles was compared with the cross-wavelet transform, giving each
participant a 13 x 13 matrix, or 78 unique pairs, describing how strongly each
pair of joints varies in amplitude together. The article asks how that
structure is organised anatomically, how it distributes across frequency, how
consistent it is within a session, how it relates to vocal activity, and how
much individual information it carries.

**The sample is 70 independent participants.** Every array in `data/` has 70
rows, one per participant, labelled `P01` to `P70`. The labels are assigned
after a fixed random permutation, so the ordering carries no information about
recording order or provenance.

---

## 3. Getting coordinates out of your own video, with DeepLabCut

Skip this section if you only want to reproduce the published numbers: the
coordinates are already in `data/`, and `reproduce_all.py` needs nothing from
here. Read it if you want to run the method on a recording of your own.

### 3.1 The model

Pose estimation uses **DeepLabCut** with the **SuperAnimal-HumanBody**
pretrained model. "Pretrained" is the point: it needs no labelled frames of
your own, no training run, and no annotation. You give it a video and it
returns 17 keypoints per person per frame.

- Model documentation and the list of available SuperAnimal models:
  <https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html>
- DeepLabCut itself: <https://github.com/DeepLabCut/DeepLabCut>

**Cite the model as well as this article.** The SuperAnimal models are the work
of another group and the article that describes them is:

```bibtex
@article{Ye2024,
  title   = {SuperAnimal pretrained pose estimation models for behavioral analysis},
  author  = {Shaokai Ye and Anastasiia Filippova and Jessy Lauer and
             Steffen Schneider and Maxime Vidal and Tian Qiu and
             Alexander Mathis and Mackenzie Weygandt Mathis},
  journal = {Nature Communications},
  year    = {2024},
  preprint= {abs/2203.07436}
}
```

### 3.2 How the analysis in the article was done

Installation instructions, the inference call and its options belong to
DeepLabCut and are documented, and kept current, by its own authors. Follow
the ModelZoo page linked above rather than a copy of it here, which would go
stale.

What was done for this study, in outline:

1. Each recording was cut into 60-second pieces with `ffmpeg`, using a stream
   copy so the frames stay bit-identical to the source. Whole 15-minute
   recordings at 60 fps do not fit in the memory of a modest GPU.
2. Each piece was passed through `video_inference_superanimal` with the
   `superanimal_humanbody` model, with test-time video adaptation enabled and
   a keypoint threshold of 0.4. Batch sizes of one throughout, again for
   memory.
3. The per-piece coordinate files were concatenated back into one continuous
   series per recording, with a cumulative frame offset so the index stays
   gap-free.

Section 2.3 of the article gives the version numbers, the exact call and the
measured effect of the clip boundaries on the joint signals. One detail is
worth repeating here because it is visible in the files: the model that
actually ran was **RTMPose-x** with a **Faster R-CNN MobileNetV3-Large-FPN**
detector. Any pose file states its own model in the first line, in the
`scorer` row. Trust that row.

### 3.3 How long it takes

Measured on the machine used for the article:

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 GB |
| System | Windows 11 Pro, Python 3.10, CUDA 11.8 |
| Batch size | 1, for detector, pose and adaptation |
| Video | 1920 x 1080, 60 fps, two people in frame |
| **Speed** | **about 12 minutes of computing per minute of video** |

So, on that hardware:

| Recording | Expected time |
|---|---|
| 1 minute | ~12 minutes |
| 5 minutes | ~1 hour |
| **15 minutes (as in the article)** | **~3 hours** |
| the ~9 hours of video behind this study | ~4 days of continuous processing |

This is not a fast operation, and it is the slowest part of the whole method by
a wide margin: once the coordinates exist, the entire wavelet analysis of 70
participants runs in about five minutes. Plan accordingly, and note that the
figure scales with resolution, frame rate and the number of people in frame.
A newer GPU with more memory will allow larger batch sizes and cut it
substantially.

### 3.4 From DeepLabCut to the viewer

DeepLabCut writes `.h5` and `.csv`. The analysis code reads either; **the
visualizer reads the `.csv`**, because it is a web page and a browser has no
HDF5 reader.

If a merge script left you with only `.h5`, convert it:

```bash
python tools/dlc_h5_to_csv.py MYVIDEO_superanimal_humanbody_....h5
python tools/dlc_h5_to_csv.py path/to/folder/      # or a whole folder
```

Then open the `.csv` with **Your data** in the visualizer. The file must have
the standard DeepLabCut layout: four header rows (`scorer`, `individuals`,
`bodyparts`, `coords`) and one row per frame, with 17 keypoints for at least
two detected individuals.

---

## 4. The data

| File | Contents |
|---|---|
| `joint_angles_15hz.npz` | Conditioned joint-angle series, `(70, 13, 12600)`, float32. The exact input to every population-level analysis. |
| `comodulation_matrices.npz` | Session-averaged cross-wavelet modulus, `(70, 13, 13)`. |
| `coherence_matrices.npz` | Wavelet coherence, `(70, 13, 13)`, bounded 0 to 1. |
| `surrogate_null_matrices.npz` | Observed and circular-shift null matrices, 100 shifts per participant. |
| `window_profiles.npz` | The same matrices computed in seven consecutive 2-minute windows. |
| `band_power.csv` | Power in the postural, gesture and micro bands, one row per participant. |
| `anthropometry.csv` | Twelve median segment lengths and thirteen mean joint angles, per half-session. |
| `pair_assignment.csv` | Each of the 78 pairs with its anatomical category and shared-keypoint count. |
| `angle_definitions.csv` | The keypoint triplet defining each of the 13 angles. |
| `participants.csv` | Participant label and sex. |

The angle series are normalized per session to 0-100% of each angle's own
range, then block-averaged from 60 fps to 15 Hz. Recordings differ slightly in
length, so rows are padded with `NaN`; take `angles[i, :, :n_samples[i]]`.

**Sequence in every 13 x 13 matrix** (rows and columns alike):

```
0 L.Elbow      3 R.Shoulder   6 L.Knee        9 R.Distal arm  12 Head/Neck
1 R.Elbow      4 L.Hip        7 R.Knee       10 L.Distal leg
2 L.Shoulder   5 R.Hip        8 L.Distal arm 11 R.Distal leg
```

### What is not here, and what these data can and cannot reveal

The source video is not part of this repository and is not redistributable.
Participants consented to analysis, not to public release of identifiable
footage. The recordings remain available from the corresponding author under a
data-sharing agreement.

The derived data carry no name, no image, no date and no study identifier.
Participants are numbered `P01` to `P70` after a fixed random permutation, so
the ordering says nothing about who was recorded when.

Two points deserve to be stated plainly rather than assumed.

The co-modulation profiles **are** individually distinctive: the article shows
that 28.6% of participants can be matched between the two halves of their own
recording. That is a statement about matching a profile to another profile of
the same session. It is not a route from these files to a person, because
there is no identity in these files to match against, and the video that would
supply one is not shared.

`anthropometry.csv` holds twelve median segment lengths and thirteen mean
joint angles per half-session. The lengths are in **pixels**, not centimetres,
and depend on where the camera stood, so they are not body measurements that
could be compared against any external record.

Anyone using these files agrees not to attempt to re-identify participants,
and not to combine them with other sources for that purpose.

---

## 5. Three things worth knowing before reading the numbers

**The cross-wavelet modulus measures simultaneous amplitude, not phase
locking.** Because `|W_a conj(W_b)| = |W_a| |W_b|`, the modulus is a product of
two marginal amplitudes with the phase difference discarded: perfect phase
locking and uniformly random phase give the same value, and joints that simply
move a lot score high. The article therefore calls the quantity *amplitude
co-modulation*, and reports wavelet coherence separately as the normalized
control (`step3`).

**Pairs of angles built from shared keypoints are not independent.** Angle 8
(`L.Distal arm`) is measured on the same three keypoints as angle 0
(`L.Elbow`), so the two are alternative parameterizations of one measurement
rather than two degrees of freedom. More generally, angles sharing keypoints
inherit correlated tracking error, which on its own would manufacture apparent
within-limb co-modulation. `step2` stratifies the 78 pairs by shared-keypoint
count for exactly this reason, and the article rests its anatomical claim on
the stratum where no keypoints are shared.

**The session average runs over the whole time-frequency plane.** The cone of
influence is computed and is shaded in the single-session displays, but it
does not filter the averages. Restricting the average to the cone lowers every
cell by roughly a tenth and leaves the category ordering, the omnibus test and
all six pairwise comparisons unchanged; `step1` computes both and prints the
comparison, so this can be checked rather than taken on trust.

---

## 6. The steps

| Script | What it establishes |
|---|---|
| `step1_comodulation_and_categories.py` | Rebuilds the matrices from the angle series, checks them against the published values, and reproduces the category statistics of Table 3 and Figure 2c. Also reports the cone-of-influence sensitivity. |
| `step2_surrogate_and_landmark_controls.py` | The circular-shift null and the stratification by shared keypoints, which together decide how much of the structure is coordination and how much is correlated measurement error. |
| `step3_coherence_control.py` | Wavelet coherence, the normalized control, and why the modulus remains the primary measure. |
| `step4_identification.py` | Within-session identification, the anthropometric baseline that outperforms it, and what survives once anthropometry is regressed out. |
| `step5_temporal_consistency.py` | Within-session temporal consistency with the participant as the inferential unit, plus reliability at the level of a single pair. |
| `step6_sex_comparison.py` | The comparison between women and men, one observation per participant. |

Each writes a JSON summary to `results/`.

---

## 7. Reusing the library

`intrabody` is small enough to read in a sitting and is usable on its own:

```python
from intrabody.wavelet import morlet_cwt, cross_wavelet, wavelet_coherence

dt = 4 / 60                       # 15 Hz
a = morlet_cwt(signal_a, dt)
b = morlet_cwt(signal_b, dt)
modulus = cross_wavelet(a, b)["Xp"]     # amplitude co-modulation
coherence = wavelet_coherence(a, b)["coh"]
```

`intrabody.categories` holds the angle definitions, the category assignment
and the shared-keypoint counts. `intrabody.statistics` holds the FDR
correction, Wilson intervals, Fisher-z transformation, effect sizes and the
identification routines.

The wavelet implementation follows Torrence and Compo (1998) with
`omega_0 = 6`; coherence follows Grinsted et al. (2004).

---

## 8. The visualizer

`viewer/IntraBodyCoModulationViewer.html` opens in any current browser by
double-clicking it. Nothing is installed, nothing is uploaded, and every
computation runs in the page.

### The bundled examples are 120 seconds. The full sessions are beside them.

Three 120-second excerpts are stored **inside** the HTML file, which is why
the menu offers them the moment the page opens. Each is the record of one
participant, taken from three different people and from three different points
in their sessions. They are display excerpts, not the material the article
analyses.

They are short for a mechanical reason. A full session is seven times longer:
embedding one would make the browser decompress and transform more than fifty
thousand frames before showing anything.

One thing was done to them that was not done to the analysed data: the
coordinates are written to two decimals. DeepLabCut prints a float32 as
something like `332.4201354980469`, and those trailing digits are an artifact
of the conversion, not measurement. At a hundredth of a pixel nothing visible
is lost, and each row drops from about 730 characters to 320, which is what
the browser has to scan for every frame it draws.

### Where each bundled file comes from

They are not anonymous samples: each is a stated stretch of a stated
participant, so that anything seen in the viewer can be traced back to the
published arrays. `P07`, `P28` and so on are the same labels used throughout
`data/`.

| File | Participant | Stretch |
|---|---|---|
| `Example_dataset_1.csv` | P07 | 120 s, starting 120 s into the analysed segment |
| `Example_dataset_2.csv` | P28 | 120 s, starting 300 s in |
| `Example_dataset_3.csv` | P54 | 120 s, starting 480 s in |
| `Full_session_1.csv` | P41 | the analysed segment in full, 52,200 frames |

The analysed segment is what the article uses: it begins 30 s (1,800 frames)
into the recording, since 30 s are trimmed from each end. Every excerpt is
taken from inside it, so what the viewer draws corresponds frame for frame
with the series in `data/joint_angles_15hz.npz` for that participant, after
the block averaging described in Section 2.5 of the article. The frame index
is restarted at zero in each excerpt; leaving the original number would say
which minute of the recording it came from.

The per-participant pose files these were cut from are not distributed. They
are coordinates of an identifiable person's body over fifteen minutes, which
is a different kind of data from the joint-angle series in `data/`, and they
were not part of what participants consented to release.

### The voice channel, and why ours is empty

The **VOICE** panel plots an audio amplitude envelope against the joint
signals on a shared time cursor. It is always available, and **LOAD VOICE CSV**
in the sidebar opens one:

```
time_s,envelope
0.00,135.1
0.05,194.3
```

Two columns and any units, at any sampling rate: the viewer reads the
rate from the time column. A single column of values also works, and is then
stretched over the duration of the pose file. `viewer/voice_envelope_TEMPLATE.csv`
is an empty file in the right format. No audio is decoded in the browser: what
the panel draws is an envelope you have already computed.

**No voice signal from this study is distributed.** The recordings do carry
room audio, and the article analyses it, but the participants consented to
analysis of their recordings and not to release of their voices.

So that the panel can still be seen working, every bundled example carries a
**demonstration envelope**: a clean synthetic tone at 0.25 Hz, generated in
the page rather than shipped as a file, and labelled as such across the panel
itself. Nobody can mistake a pure tone for speech, which is the point. Load
your own envelope to replace it.

Two further points about the audio in the article, which apply to any envelope
you supply yourself: it is a single channel for the room, not one per speaker,
so it cannot attribute sound to either person; and the threshold used in the
article separates epochs of high from low acoustic activity, not speech from
verified silence.

That fourth excerpt is on disk as a pose file like any other:

```
viewer/Example_dataset_4.csv        pose, 7,200 frames at 60 fps
```

So the full sessions are supplied **as separate files** too:

```
viewer/Full_session_1.csv     one participant, 52,200 frames at 60 fps
```

To open one, click **Your data** in the viewer and select the file. It is the
same DeepLabCut format the viewer already reads, so nothing else changes:
the skeleton, the joint-angle traces, the scalogram and the 13 x 13 matrix all
work exactly as with the bundled examples, but over the whole session that the
article analyses. Loading takes longer, because every wavelet transform is
computed up front.

The full session starts where the analysis starts: the source videos run 15
minutes and 30 seconds are trimmed from the beginning, so frame 0 of this file
is frame 0 of the published time series.

### The sidebar, and what it selects

Every file distributed here holds **one participant**, so there is nothing to
choose and no selector appears. If you load a file that contains several
individuals, which is what DeepLabCut writes when more than one person is in
frame, a **Position in frame** control shows up and the viewer draws one at a
time. It is a property of the file format, not of the study.

The 17 checkboxes under **Keypoints** are the tracked landmarks. The 13 joint
signals the article reports are derived from them, each measured at one
keypoint as its vertex, which is why unchecking a keypoint also hides the
signal measured there. The counter under the list says how many of the 13 are
currently drawn.

Joint signals measured at distal keypoints are named *distal arm* and *distal
leg*: the 17-keypoint skeleton has no hand or foot landmark, so they describe
limb-segment alignment rather than wrist or ankle flexion.

Neither the excerpts nor the full sessions carry a participant identifier or a
recording date.

### Getting the numbers out, not just the picture

The **SAVE** button offers two formats for whichever panel is open. **PNG**
writes the panel as drawn. **CSV** writes the numbers behind it, restricted to
the time range currently on screen, with a short header naming the dataset,
the participant, the time range and the sampling rate:

| Panel | What the CSV contains |
|---|---|
| ANGLES | one column per visible joint angle, in degrees, against time |
| NORM | the same series after min-max normalisation to 0-100 |
| VOICE | the envelope and the mean joint angle, against time |
| MATRIX | the 13 x 13 matrix, cross-wavelet power or coherence |
| WAVELET | one row per frequency, with its mean power over the window shown |

The files open directly in any spreadsheet. The wavelet panel is summarised by
frequency rather than written out in full, because a scalogram at display
resolution is thousands of columns wide and no spreadsheet handles that
usefully; the complete matrices are in `data/` for anyone who needs them.

### What the three columns per keypoint contain

Both the excerpts and the full sessions are unfiltered SuperAnimal-HumanBody
output, the file a user obtains by running the model on their own video. The
coordinates are in pixels of the original frame and are given to two decimals;
frames in which the detector returned no pose are left empty rather than
filled.

The third column is the model's keypoint score. **It is not a probability and
it is not bounded at 1**: the RTMPose head reports the peak response of its
coordinate classifier, which is unnormalized, so roughly one value in six lies
between 1.0 and 1.4. Higher still means more confident, and the values are
comparable within a file. They are reproduced exactly as the model wrote them;
truncating them at 1 would discard the ordering among the most confident
detections. Any threshold applied to them should be chosen from the
distribution in the file rather than read as a probability.

One step is applied on top of the raw output. The detector proposes up to ten
candidate individuals per frame, and its `animal0`/`animal1` labels move
between people over the course of a recording. The files distributed here were
split beforehand, **one participant per file**, using the assignment described
in Section 2.4 of the article; the split was checked against the coordinates
the published pipeline stored and agrees with them to 0.00 pixels. That is why
no side selector appears: there is only one person in the file. The empty
candidates are
dropped. Across all 17 keypoints of both files, one frame out of 50,400
crosses between the two people.

The skeleton overlay, the joint signals and the voice channel advance on a
common time cursor, and the cross-wavelet surface of any pair can be shown
alongside them. Joint signals and the 13 x 13 matrix are derived in the
browser when a file is loaded, and a scalogram is computed when its panel or
its pair is selected; playback itself only redraws. No pose estimation happens
here at any point: that is the separate DeepLabCut step described in Section 3.

---

## 9. Licence and how to cite

Code under `code/` and `viewer/`: **MIT**.
Data under `data/`: **Creative Commons Attribution 4.0 International**.

Both licences permit reuse, including commercially, and both require that the
copyright notice and the `NOTICE` file be kept with any copy, substantial
portion or derivative work.

### How to cite

Cite the article, not the repository:

> Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. Interactive playback
> visualizer to analyze joint-angle co-modulation with a wavelet approach:
> application to pose-voice relationships during spontaneous conversation.
> *Machine Learning and Knowledge Extraction*, 2026.

`CITATION.cff` carries the same reference in machine-readable form, so most
reference managers and repository interfaces can import it directly.

The citation also travels with the files themselves, in case one of them is
ever separated from this directory: every script carries it in its header,
every `.npz` carries it in a `citation` field, every data CSV opens with it as
a comment line, and the viewer displays it in a footer.

```python
>>> import numpy as np
>>> str(np.load("data/comodulation_matrices.npz", allow_pickle=True)["citation"])
'Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. Interactive playback ...'
```
