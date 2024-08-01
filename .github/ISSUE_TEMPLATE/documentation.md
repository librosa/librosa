---
name: Documentation
description: Create a report to help us improve the documentation
title: "[Doc]: "
labels: [Documentation]
body:
  - type: input
    id: link
    attributes:
      label: Documentation Link
      description: >-
        Link to any documentation or examples that you are referencing. Suggested improvements should be based
        on [the development version of the docs](https://librosa.org/doc/main)
      placeholder: https://librosa.org/doc/main/...
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What is missing, unclear, or wrong in the documentation?
      placeholder: |
        * I found [...] to be unclear because [...]
        * [...] made me think that [...] when really it should be [...]
        * There is no example showing how to do [...]
    validations:
      required: true
  - type: textarea
    id: improvement
    attributes:
      label: Suggested improvement
      placeholder: |
        * This line should be be changed to say [...]
        * Include a paragraph explaining [...]
        * Add a figure showing [...]
