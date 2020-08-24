---
title: "Data Science Workflow"
toc: true
layout: post
categories: [data science, machine learning, kedro]
image: "/images/data_science_workflow/gitflow.jpg"
---

I don't come from a software engineering background. Most data scientists I know, also don't.  

We can argue that some of our work will never be executed again and we shouldn't waste time organizing it. The big pletora of tools introduced every day may overwhelm even the workaholics. This may lead to analysis paralysis and, for me, this is worse than doing things in an unorganized manner.

# A method to the madness

I've been testing different methods to better organize my data science work. The main objectives for me are:

- Ensure **reproducibility** of my work
- Be **modular** in order to be maintanable and extensible
- Generate **documentation** without hassle

#### Disclaimer

This is a work in progress and may be updated with time.

## OS

Unix based operating systems, so is either Mac or some Linux distribution. Ubuntu works without much hassle, the latest 19.04 with the Gnome patches is working well so far.

## Python

Python 3, for Data Science you can save a lot of time by installing [Anaconda](https://www.anaconda.com/distribution/#linux).

## Python virtual environment

Using conda you can create a python 3.7 environment called *env_name* by running:

```
conda create --name env_name python=3.7
```

## Creating a new project

I started using [Kedro](https://kedro.readthedocs.io/en/latest/index.html) lately to structure my projects.

You can use others like the great [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science).

Open the terminal and navigate to the directory in which you want to work. Activate your environment created on the previous step by running:

```
conda activate env_name
```

Install kedro:

```
pip install kedro
```

Create a new project:

```
kedro new
```

A new folder will be created in `<current_dir>/<repo_name>/`

Navigate to this folder `cd <repo_name>` and move the credentials.yml file to the local configuration:

```
mv ./conf/base/credentials.yml ./conf/local/
```

## Data

The workflow starts to become more specific now. If you have the data as file(s), copy it to the *data* folder in the correct subfolder. If you are querying from a database, read Kedro [documentation](https://kedro.readthedocs.io/en/latest/04_user_guide/04_data_catalog.html) for guidelines.

Reference all datasets for the project in `conf/base/catalog.yml`

## Exploration

Use the notebooks folder to explore and draft experimental code in Jupyter notebooks. Once you have a good understanding of the data and have written your functions to process it, copy them to `src/<project_name>/nodes/` as `.py` file(s).

## Nodes

Create the data transformation steps as [pure](https://en.wikipedia.org/wiki/Pure_function) Python functions when possible. With every step coded as functions in the nodes folder, it's time to create the pipeline.

## Pipeline

The data pipeline code goes in `src/<project_name>/pipeline.py`. You need to specify the inputs and outputs of the functions. Make use of the `conf/base/catalog.yml` to document the datasets.

Finally, choose if you want to run the pipeline in sequence or in parallel (`--parallel` on the command below).

```
kedro run
```

## Documentation

Include `docstrings` to explain your Python functions, so you can take advantage of [auto-generated Sphinx documentation](http://www.sphinx-doc.org/en/master/).

Build the project documentation:
```
kedro build-docs
```

## Other details

I simplified the process here, I recommend that you run both examples on the Kedro documentation page to get used to the workflow. 

Use [Git](https://git-scm.com/) to version control your codebase.

Ideally, you should also write **unit tests** and add **CI** configuration to your repository.


## Conclusion

Quoting the main features of Kedro that won me over:

> Kedro is a workflow development tool that helps you build data pipelines that are robust, scalable, deployable, reproducible and versioned. We provide a standard approach so that you can:
> 
> - spend more time building your data pipeline,
> - worry less about how to write production-ready code,
> - standardise the way that your team collaborates across your project,
> - work more efficiently.

If you are starting with data science, don't over organize things. Use simple tools to understand the concepts and play with models/visualization. Have fun playing around, and you are more likely to want to advance on complex concepts.