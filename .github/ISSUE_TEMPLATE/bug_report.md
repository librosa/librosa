---
name: Bug Report
description: Report a bug or issue with librosa.
title: "[Bug]: "
body:
  - type: textarea
    id: summary
    attributes:
      label: Bug summary
      description: Describe the bug in 1-2 short sentences
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Code for reproduction
      description: >-
        If possible, please provide a minimum self-contained example.
      placeholder: Paste your code here. This field is automatically formatted as Python code.
      render: Python
    validations:
      required: true
  - type: textarea
    id: actual
    attributes:
      label: Actual outcome
      description: >-
        Paste the output produced by the code provided above, e.g.
        console output, any relevant screenshots/screencasts, etc.
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected outcome
      description: Describe (or provide an example of) the expected outcome from the code snippet.
    validations:
      required: true
  - type: textarea
    id: details
    attributes:
      label: Additional information
      description: |
        - What are the conditions under which this bug happens? input parameters, edge cases, etc?
        - Has this worked in earlier versions?
        - Do you know why this bug is happening?
        - Do you maybe even know a fix?
  - type: input
    id: operating-system
    attributes:
      label: Operating system
      description: Windows, OS/X, Arch, Debian, Ubuntu, etc.
  - type: textarea
    id: software-versions
    attributes:
      label: software versions
      description: >-
        Please run the following Python code snippet and paste the output here:
        
        import platform; print(platform.platform())
        import sys; print("Python", sys.version)
        import numpy; print("NumPy", numpy.__version__)
        import scipy; print("SciPy", scipy.__version__)
        import librosa; print("librosa", librosa.__version__)

        librosa.show_versions()
        
    validations:
      required: true
  - type: dropdown
    id: install
    attributes:
      label: Installation
      description: How did you install librosa?
      options:
        - pip
        - conda
        - git checkout
        - unable to install
