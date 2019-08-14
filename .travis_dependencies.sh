#!/bin/sh

ENV_NAME="test-environment"
set -e

conda_create ()
{

    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda config --add channels pypi
    conda info -a
    deps='pip numpy scipy coverage scikit-learn!=0.19.0 matplotlib numba'

    conda create -q -n $ENV_NAME "python=$TRAVIS_PYTHON_VERSION" $deps
    conda update --all
}

src="$HOME/env/miniconda$TRAVIS_PYTHON_VERSION"
if [ ! -d "$src" ]; then
    mkdir -p $HOME/env
    pushd $HOME/env

        # Download miniconda packages
        if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
            wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
        else
            wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
        fi

        # Install both environments
        bash miniconda.sh -b -p $src

        export PATH="$src/bin:$PATH"
        conda_create

        source activate $ENV_NAME

        conda install -c conda-forge ffmpeg pysoundfile python-coveralls

        if [[ "$TRAVIS_PYTHON_VERSION" == "3.5" ]]; then
            conda install -c numba "numba>=0.43"
        fi

        source deactivate
    popd
else
    echo "Using cached dependencies"
fi
