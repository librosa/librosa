# Compiling librosa documentation

For a single-version build, we can use the Makefile in the `docs/` folder, e.g.:

```
make html
```
or
```
make latexpdf
```

For a multi-version historical build (i.e., our doc site), we need to use
sphinx-multiversion.

This can be done from the top-level source directory of the repository as follows:

```
sphinx-multiversion docs/ build/html
```

This says that the source config lives in `docs/` and the output site will be
deposited under `build/html`.  To deploy, we sync the compiled site to the
`librosa-doc` repository, which in turn publishes to github-pages.

## Notes

Because the historical docs include example code that is executed to generate
figures, the environment for building historical docs can be brittle.
Presently, the oldest compiled doc is for release 0.6.3.
The historical docs work with the following dependency versions:

    - numba=0.48 : decorators submodule move in 0.49 gives warnings, and 0.50 breaks old librosa
    - numpy=1.17 : strict dtype requirements in linspace parameters break some of our old examples from 1.18 on
    - matplotlib=3.2 : log axes API changes cause warnings in 3.3 onward
