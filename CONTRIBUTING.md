
Contributing code
=================

How to contribute
-----------------

The preferred way to contribute to librosa is to fork the 
[main repository](http://github.com/librosa/librosa/) on
GitHub:

1. Fork the [project repository](http://github.com/librosa/librosa):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

   ```shell
   git clone git@github.com:YourLogin/librosa.git
   ```

3. Set the upstream remote to the Librosa's repo:

   ```shell
   git remote add upstream git@github.com:librosa/librosa.git
   ```

4. Create a new conda environment in order to install dependencies:

   ```shell
   conda create -n librosa-dev python=3.13
   conda env update -n librosa-dev --file .github/environment-ci.yml
   conda activate librosa-dev
   python -m pip install -e '.[tests]'
   ```

5. Create a branch to hold your changes:

   ```shell
   git switch -c <NAME-NEW-BRANCH>
   ```

   and start making changes. Never work in the ``main`` branch!

6. Work on this copy on your computer using Git to do the version
   control. You can check your modified files using:

   ```shell
   git status
   ```

7. When you're done editing, do:

   ```shell
   git add <PATH-TO-MODIFIED-FILES>
   git commit -m "<COMMIT-MESSAGE>"
   ```

   to record your changes in Git, then push them to GitHub with:

   ```shell
   git push --set-upstream origin <NAME-NEW-BRANCH>
   ```

8. Go to the web page of your fork of the librosa repo,
   and click 'Pull request' to review your changes and add a description
   of what you did.

Finally, click 'Create pull request' to send your changes to the
maintainers for review. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](https://git-scm.com/docs) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

- All functions should have informative [docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) with sample usage presented.

You can also check for common programming errors with the following
tools:

- Code with good test coverage, check with:

  ```shell
  pytest
  ```

-  Style adherence, check with:

    ```shell
    conda install ruff
    ruff check librosa
    ```

- Ensure that your docstrings are properly formatted, check with:

  ```shell
  python -m pip install velin
  python -m velin --check librosa
  ```

- Ensure that any new functionality has valid type annotations, check with:

  ```shell
  python -m pip install mypy
  python -m mypy librosa
  ```

Some tests in tests/test_display.py use baseline images for output comparison.
If existing images need to be updated or new ones should be added:

1. Ensure that the environment is properly setup for testing (pytest with addons)
2. Run:

   ```shell
   pytest --mpl-generate-path=tmp tests/test_display.py [-k ANY_TESTS_THAT_CHANGED]
   ```

3. Inspect the new baseline images under tmp/
4. If (3) looks good, copy into `tests/baseline_images/test_display/` and add to the PR.

Additionally, some functions require network access to test properly.
If you are adding tests that require network access, please mark them with `@pytest.mark.network`.

Filing bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

- Verify that your issue is not being currently addressed by other
  [issues](https://github.com/librosa/librosa/issues?q=)
  or [pull requests](https://github.com/librosa/librosa/pulls?q=).

- Please ensure all code snippets and error messages are formatted in
  appropriate code blocks.
  See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

- Please include your operating system type and version number, as well
  as your Python, librosa, numpy, and scipy versions. This information
  can be found by running the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import librosa; print("librosa", librosa.__version__)
  ```

Documentation
-------------

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the docs/ directory.
The resulting HTML files will be placed in _build/html/ and are viewable 
in a web browser. For detailed instructions on building the documentation, see [docs/BUILD.md](https://github.com/librosa/librosa/blob/main/docs/BUILD.md).

For building the documentation, you will need some additional dependencies.
These can be installed by executing the following command:

```shell
python -m pip install -e '.[docs]'
```

Policy on Using AI Assistance (Copilot, Gemini, ChatGPT, etc.)
--------------------------------------------------------------

We do not micromanage your development workspace, nor do we require you to track every prompt or model version used while brainstorming. If you use generative AI as an advanced search engine, a pair programmer, a rubber duck to debug tricky math, or a helper to build a boilerplate utility script, you are completely free to do so.

Instead of administrative tracking, we ask that you follow these principles in developing contributions:

### 1. The Accountability and Explainability Standard

You are the author of your Pull Request, not the AI.
If a maintainer asks why your code is the way it is, replying "that's just what the AI produced" is not acceptable. You must completely understand, be able to explain, and technically stand behind every line of code you submit.

### 2. The Verification Imperative

If an AI assistant was used to generate code blocks, data parsers, regular expressions, or core signal processing logic, ensure that the solution falls into a category where correctness can be confidently and efficiently verified.

### 3. No AI-Generated Repository Dialogue

While AI is welcome in your private code-generation loop, it is not welcome in our community dialogue space.

We (the maintainers) do reserve the right to use our own automation and AI tools in
generating issues, pull requests, or comments.
We **do not** invite external AI contributions of this form however, as they
generate a significant amount of work for us maintainers to process.

Note
----
This document was gleefully adapted from [scikit-learn](http://scikit-learn.org/).

The AI policy was generated in part by Gemini 3.5 Flash as a synthesis of similar
policies indexed by [@melissawm](https://github.com/melissawm/open-source-ai-contribution-policies).  
Yes, we appreciate the irony in that, and the initial draft has been heavily revised and edited by human maintainers.
