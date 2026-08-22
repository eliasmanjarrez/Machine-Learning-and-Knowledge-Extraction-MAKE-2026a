# Intra-body joint-angle co-modulation: analysis pipeline and derived data

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

**Nothing here reads video.** Everything begins with a table of body
coordinates, and producing that table is a separate step that you run
yourself, with DeepLabCut. The visualizer reads such a table directly; the
analysis scripts start one step further along, from the joint angles already
derived from those coordinates and deposited in `data/`.

```
   your video (.mp4)
        │
        │   DeepLabCut + SuperAnimal-HumanBody       ← you run this, once
        ▼
   pose coordinates (.h5 and .csv)
        17 keypoints per person per frame,
        three columns each: x, y, likelihood
        │
        │   this repository, or the visualizer in your browser:
        │   three keypoints define one angle, measured at the middle one
        ▼
   13 joint angles per person, one value per angle per frame
        │
        │   wavelet analysis
        ▼
   figures, tables, exported data
```

Point the visualizer at an `.mp4` and nothing will happen: it is a web page,
and it has no pose-estimation model inside it. Section 3 says which model to
use, where its documentation lives, how the analysis in the article was run,
and how long it takes.

### Coordinates and angles are never the same file

This is the distinction that is easiest to lose, so it is worth stating
flatly: **what DeepLabCut writes are coordinates, not angles.**

A pose file is the output of the pose-estimation model. One row is one video
frame, and each of the 17 keypoints occupies three columns, `x`, `y` and
`likelihood`: two positions in pixels of the original frame, and the score the
model assigned to them. Such a file contains no angle of any kind. The pose
files distributed here are the four `.csv` in `viewer/`, and they are what the
visualizer opens.

A joint-angle series is what the analysis works on. One row is one time
sample, each of the 13 columns is one joint angle, and the values are degrees,
or, after the conditioning described in Section 4, percentages of each angle's
own range. These series are in `data/joint_angles_15hz.npz`. They are computed
from the coordinates; no pose-estimation model produces them.

The step between the two is arithmetic on triplets of keypoints: three
keypoints define one angle, measured at the middle one, so that
`left_shoulder` - `left_elbow` - `left_wrist` gives the left elbow angle.
`data/angle_definitions.csv` lists the thirteen triplets in full. Hence `data/`
holds angles and no coordinates, `viewer/` holds coordinates and no angles, and
the visualizer performs this step in the browser every time a file is opened
(Section 8).

---

## 1. What is here, and what it is for

The repository lets a reader recompute every quantity the article reports, from
the conditioned joint-angle series through to the last p-value, without access
to the source video.

```
README.md            this file
LICENSE              MIT for the code, CC-BY-4.0 for the data
CITATION.cff         machine-readable citation
NOTICE               the attribution that travels with any copy

code/                the analysis pipeline
  intrabody/         the library: wavelet core, anatomy, statistics
  step1 ... step6    one script per analysis, each printing its own numbers
  reproduce_all.py   runs all six and reports what matched

data/                de-identified derived data: the joint angles and the
                     quantities computed from them, 40 MB in total
viewer/              the visualizer, four pose files, a voice template
tools/               dlc_h5_to_csv.py, to feed DeepLabCut output to the viewer
```

The quickest way to see whether the code works:

```bash
cd code
python reproduce_all.py
```

That takes about five minutes. Every step prints its recomputed values beside
the values printed in the article, and the final line counts how many matched:
26 values are checked, and all 26 match.
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
derived joint-angle series are already in `data/`, and `reproduce_all.py` needs
nothing from here — no video, no coordinates and no DeepLabCut installation.
Read this section only if you want to run the method on a recording of your
own, which is the one case where you have to produce the coordinates yourself.

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
`bodyparts`, `coords`) and one row per frame, with 17 keypoints. One individual
per file is the ordinary case, and is what every file distributed here
contains. A file holding several is read as well: a **Position in frame**
control then appears and the visualizer draws one person at a time.

---

## 4. The data

Everything here is derived from the pose coordinates; none of it is
coordinates. The pose files are in `viewer/`, and Section 8 describes them.

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
the ordering says nothing about who was recorded when. The same holds for the
four pose files in `viewer/`, which are tables of pixel coordinates and are
described in Section 8.

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

### The bundled examples are 120 seconds. One full session sits beside them.

Three 120-second excerpts are stored **inside** the HTML file, compressed,
which is why the menu offers them the moment the page opens. Each is the record
of one participant, taken from three different people and from three different
points in their sessions. They are display excerpts, not the material the
article analyses.

They are short for a mechanical reason. A full session is seven times longer:
embedding one would make the browser decompress and transform more than fifty
thousand frames before showing anything. The one full session that is
distributed therefore sits **beside** the page as an ordinary file,
`viewer/Full_session_1.csv`, and is opened with **Your data** like any file of
your own.

One thing was done to them that was not done to the analysed data: the
coordinates are written to two decimals. DeepLabCut prints a float32 as
something like `332.4201354980469`, and those trailing digits are an artifact
of the conversion, not measurement. At a hundredth of a pixel nothing visible
is lost, and each row drops from about 730 characters to 320, which is what
the browser has to scan for every frame it draws.

### Where each file in `viewer/` comes from

They are not anonymous samples: each is a stated stretch of a stated
participant, so that anything seen in the viewer can be traced back to the
published arrays. `P07`, `P28` and so on are the same labels used throughout
`data/`.

| File | Participant | Stretch |
|---|---|---|
| `Example_dataset_1.csv` | P07 | 120 s, starting 120 s into the analysed segment |
| `Example_dataset_2.csv` | P28 | 120 s, starting 300 s in |
| `Example_dataset_3.csv` | P54 | 120 s, starting 480 s in |
| `Full_session_1.csv` | P41 | 52,200 frames, 14.5 min: the recording from 30 s in to its end |

The **analysed segment** is what the article uses: it begins 30 s (1,800
frames) into the recording and ends 30 s before it finishes, because 30 s are
trimmed from each end. At 60 fps that is 50,400 frames, which after the block
averaging described in Section 2.5 of the article is the 12,600 samples per
participant held in `data/joint_angles_15hz.npz`.

`Full_session_1.csv` starts where the analysed segment starts, so its frame 0
is sample 0 of the published series for P41, and it runs 1,800 frames past the
end of that segment: the last 30 s of the file are the tail the analysis
trims. The three excerpts are taken from inside the analysed segment, at the
offsets stated in the table.

Every excerpt therefore covers the same stretch of the same recording as the
corresponding samples of `data/joint_angles_15hz.npz`, but it is not the same
series: the analysis block-averages 60 fps to 15 Hz, whereas the visualizer
displays every sixth frame, at 10 Hz, and normalises each file over its own
length rather than over the whole session. The movement is the same; the
sampling and the scaling are not. The frame index is restarted at zero in each
excerpt; leaving the original number would say which minute of the recording
it came from.

What is published and what is not, plainly. The source video is not published
and neither is the audio. Of the seventy per-participant pose files, this
repository distributes one in full (`Full_session_1.csv`) together with three
120-second excerpts, as the material the visualizer needs in order to be
reproducible without the recordings. A pose file is a table of pixel
coordinates: it carries no image, no sound, no name and no date, and the video
that would attach a face to it is not shared. The recordings themselves remain
available from the corresponding author under a data-sharing agreement.

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

So that the panel can still be seen working, each of the three embedded
examples carries a **demonstration envelope**: a clean synthetic tone at
0.25 Hz, generated in the page rather than shipped as a file, and labelled as
such across the panel itself. Nobody can mistake a pure tone for speech, which
is the point. Load your own envelope to replace it. A file you open yourself,
`Full_session_1.csv` included, starts with no envelope at all, and the panel
then says how to load one.

Two further points about the audio in the article, which apply to any envelope
you supply yourself: it is a single channel for the room, not one per speaker,
so it cannot attribute sound to either person; and the threshold used in the
article separates epochs of high from low acoustic activity, not speech from
verified silence.

### Opening a file takes time, and the page says how much

To open the full session, or any DeepLabCut file of your own, click **Your
data** and select the `.csv`. It is the same format the visualizer already
reads, so nothing else changes: the skeleton, the joint-angle traces, the
scalogram and the 13 x 13 matrix all work exactly as with the bundled
examples.

What does change is the wait. Nothing is precomputed inside the page. When a
file is opened, the 13 angle series are derived from the coordinates, and then
13 wavelet transforms and 78 cross-wavelet pairs are computed before anything
is drawn. The cost grows faster than the length of the recording, because a
longer record is also given longer wavelet scales.

Measured in Chrome on the machine used for the article, one of the 120-second
excerpts, 7,200 frames, opens in ten to fifteen seconds, and
`Full_session_1.csv`, 52,200 frames or 14.5 minutes, in about two minutes. A
slower machine takes proportionally longer, and a recording of your own scales
with its own duration rather than with its file size.

None of this has to be taken on trust while waiting. The page reports how many
of the 78 pairs are done, how long it has taken so far and roughly how much is
left, all timed on the machine in front of you rather than assumed, and once
the file is open the header states how long it actually took. A page that
seems frozen for a minute on a long recording is a page that is still
computing.

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

No file carries a participant identifier or a recording date inside it; the
table under *Where each file in `viewer/` comes from* is what states the
correspondence.

### Getting the numbers out, not just the picture

The **SAVE** button offers two formats for whichever panel is open. **PNG**
writes the panel as drawn. **CSV** writes the numbers behind it, restricted to
the time range currently on screen, with a short header naming the dataset,
the time range and the sampling rate — and, only when the file holds more than
one individual, which of them is drawn:

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

The excerpts and the full session alike are unfiltered SuperAnimal-HumanBody
output, the file a user obtains by running the model on their own video. The
coordinates are in pixels of the original frame and are given to two decimals.
A frame in which the detector returned no pose would be left empty rather than
filled; in the four files distributed here there are none.

The third column is the model's keypoint score. **It is not a probability and
it is not bounded at 1**: the RTMPose head reports the peak response of its
coordinate classifier, which is unnormalized, so between an eighth and a
quarter of the values in these files, depending on the file, lie above 1.0,
and none exceeds 1.4. Higher still means more confident, and the values are
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
no **Position in frame** selector appears: there is only one person in the
file, and the empty candidates are dropped. Across all 17 keypoints, one frame
in the 50,400 of the analysed segment crosses between the two people.

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
