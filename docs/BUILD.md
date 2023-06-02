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
sphinx-multiversion -D smv_latest_version=$(./scripts/get_latest_release.sh) docs/ build/html
```

This says that the source config lives in `docs/` and the output site will be
deposited under `build/html`.  To deploy, we sync the compiled site to the
`librosa-doc` repository, which in turn publishes to github-pages.

## Notes

Because the historical docs include example code that is executed to generate
figures, the environment for building historical docs can be brittle.
Presently, the oldest compiled doc is for release 0.8.1.

For debugging documentation builds, you can disable docstring example code
execution by setting the `LIBROSA_DOC_DEBUG` environment variable prior to
compiling the documentation.  This will significantly reduce the time needed to
build the documentation site.
