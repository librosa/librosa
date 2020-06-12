---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---


***BEFORE POSTING A BUG REPORT*** Please look through [existing issues (both open and closed)](https://github.com/librosa/librosa/issues?q=is%3Aissue) to see if it's already been reported or fixed!


**Describe the bug**
A clear and concise description of what the bug is.


**To Reproduce**
<!--
Example:
```
import librosa

# Generate one frame of random STFT-like energy
S = np.random.randn(1025, 1)**2

contrast = librosa.feature.spectral_contrast(S=S)
```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Software versions***
<!--
Please run the following Python code snippet and paste the output below.
```
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import librosa; print("librosa", librosa.__version__)

librosa.show_versions()
```
-->

**Additional context**
Add any other context about the problem here.
