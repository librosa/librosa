
.. tab-set::
    .. tab-item:: pip

        The simplest way to install *librosa* is through the Python Package Index (PyPI).
        This will ensure that all required dependencies are fulfilled.
        This can be achieved by executing the following command::

            pip install librosa

    .. tab-item:: conda

        If you use conda/Anaconda environments, librosa can be installed from the `conda-forge` channel::

            conda install -c conda-forge librosa

    .. tab-item:: source

        If you've downloaded the archive manually from the `releases
        <https://github.com/librosa/librosa/releases/>`_ page, you can install using the `setuptools` script::

            tar xzf librosa-VERSION.tar.gz
            cd librosa-VERSION/
            python setup.py install

        If you intend to develop librosa or make changes to the source code, you can
        install with `pip install -e` to link to your actively developed source tree::

            tar xzf librosa-VERSION.tar.gz
            cd librosa-VERSION/
            pip install -e .

        Alternately, the latest development version can be installed via pip::

            pip install git+https://github.com/librosa/librosa
