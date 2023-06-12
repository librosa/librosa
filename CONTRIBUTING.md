
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

          $ git clone --recursive git@github.com:YourLogin/librosa.git
          $ cd librosa 
          $ git pull --recurse-submodules

    These commands will clone the main librosa repository, as well as the submodule
    that contains testing data.  This should work automatically, but you may also
    want to read [Working with Submodules](https://github.blog/2016-02-01-working-with-submodules/)
    for a better understanding of how this works.

3. Set the upstream remote to the Librosa's repo:

          $ git remote add upstream git@github.com:librosa/librosa.git 

4. Create a new conda environment in order to install dependencies:

          $ conda create -n librosa-dev python=3.9

          $ conda env update -n librosa-dev --file .github/environment-ci.yml

          $ conda activate librosa-dev

          $ python -m pip install -e '.[tests]'

5. Create a branch to hold your changes:

          $ git switch -c <NAME-NEW-BRANCH>

   and start making changes. Never work in the ``main`` branch!

6. Work on this copy on your computer using Git to do the version
   control. You can check your modified files using:

          $ git status 

7. When you're done editing, do:

          $ git add <PATH-TO-MODIFIED-FILES>

          $ git commit -m "<COMMIT-MESSAGE>"

   to record your changes in Git, then push them to GitHub with:

          $ git push --set-upstream origin <NAME-NEW-BRANCH>

8. Go to the web page of your fork of the librosa repo,
   and click 'Pull request' to review your changes and add a description
   of what you did.

Finally, click 'Create pull request' to send your changes to the
maintainers for review. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  All functions should have informative [docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) with sample usage presented.

You can also check for common programming errors with the following
tools:

-  Code with good test coverage, check with:

          $ pytest

-  No pyflakes warnings, check with:

           $ python -m pip install flake8
           $ flake8 librosa


Some tests in tests/test_display.py use baseline images for output comparison.
If existing images need to be updated or new ones should be added:
1. Ensure that the environment is properly setup for testing (pytest with addons)
2. Run:

           $ pytest --mpl-generate-path=tmp tests/test_display.py [-k ANY_TESTS_THAT_CHANGED]

3. Inspect the new baseline images under tmp/
4. If (3) looks good, copy into `tests/baseline_images/test_display/` and add to the PR.

Finally, once your pull request has been created and reviewed, please update the file `docs/changelog.rst`
to briefly summarize your contribution in the section for the next release.
If you are a first-time contributor, please add yourself to `AUTHORS.md` as well.

Filing bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/librosa/librosa/issues?q=)
   or [pull requests](https://github.com/librosa/librosa/pulls?q=).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
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
in a web browser. See the README file in the doc/ directory for more information.

For building the documentation, you will need some additional dependencies.
These can be installed by executing the following command:

    $ python -m pip install -e '.[docs]'
    
Note
----
This document was gleefully borrowed from [scikit-learn](http://scikit-learn.org/).
