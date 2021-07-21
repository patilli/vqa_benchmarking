Installation
============

Requirements
------------

- Linux or macOS with Python >=3.6
- PyTorch and torchvision
- Node.js

Build - Inital Step
-------------------

.. code-block:: bash

    # get the latest version
    git clone https://github.com/patilli/vqa_benchmarking.git


Python Backend
--------------

The backend is written in ``Python`` for ``PyTorch``.
It provides datasets and model adapters to be integrated into any ``PyTorch`` repository.
The data gets stored in ``sqlite3`` databases with a ``tornado web server``.


Build
^^^^^

We recommend to create and activate a new python virtual environment for the following instructions.

.. code-block:: bash

    # change directory
    cd vqa_benchmarking/vqa_benchmarking_backend

    # install required libraries   
    pip install -r requirements.txt

    # install (consider adding -e)
    pip install .

Web Interface
-------------

The web application is written with ``vue.js``.
It requires a javascript package manager like ``npm`` to be installed on your machine.

Build
^^^^^

.. code-block:: bash

    # change directory
    cd vqa_benchmarking/vqa_benchmarking

    # install dependencies
    npm install

    # serve with hot reload at localhost:8080
    npm run dev

    # build for production with minification
    npm run build

    # build for production and view the bundle analyzer report
    npm run build --report
