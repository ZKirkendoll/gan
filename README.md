# Docker Boilerplate for deploying TensorFlow Apps

A simple container for deploying Python apps that depend on all that
CUDA junk that you need. This is meant to make it easy to deploy apps
with little configuration so that you can use TensorFlow with little
difficulty.

## Installation

Run

    git clone git@github.com:chadac/tf-docker-boilerplate.git [project-name]
    cd [project-name]
    rm -rf .git

to copy the boilerplate code. Then, to set up the project, simply run

    echo [project-name] >> .name

Make sure that this name contains no spaces and is composed of only
valid characters for Docker tags.

### Installing Additional Dependencies

You do not need to install dependencies directly on your system;
simply add each dependency to a new line in the `requirements.txt`
file.

## Usage

All files in the project directory (those that contain this file) are
installed to the working directory `/app` on the container. Test the
code with

    ./exec python sample.py

and everything should run correctly.
