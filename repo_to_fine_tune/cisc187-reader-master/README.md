# Intermediate Modern C++: Interactive Edition

A C++ textbook based on [runestone interactive](http://runestoneinteractive.org/build/html/index.html).

This project began with an effort to choose a suitable textbook specifically 
for students in my CISC187 C++ course.
The goals of this project are to produce a text book that:

* Covers C++ programming at the _intermediate_ level - not too basic, but not too advanced.
* Places an emphasis on Getting the most out of 'modern' C++ - that is C++11 and later.
  * There are plenty of places to learn standard C or C++98.
* Is highly interactive and supports active learning.
  * Programming is not a "spectator sport".  
    It is something you do, something you participate in. 
    It makes sense, then, that the book you use to learn programming allows you to be active.

This book is meant to provide you with an interactive experience as you learn C++.
You can read the text, watch videos, answer questions, and write and execute code.
In addition to simply executing code,
there is a unique feature called 'codelens' that allows you to control the
flow of execution in order to gain a better understanding of how the program
works.


# Using this book
If you simply want to check it out, read it or whatever,
then you're done.
You can see and read this book [online](https://daveparillo.github.io/cisc187-reader).

# Building this book from source
We have tried to make it easy for you to build and use this book.  
You can build it and host it yourself in just a few simple steps.

Use whichever of these methods works best for you.

## Install Docker and use a docker container
If you don't already have a python development environment setup,
I personally think this is easiest.
Plus I use docker for other things every day.

1. Install [Docker](https://www.docker.com/)
2. Get the Runestone docker image.

   The [repo](https://github.com/DaveParillo/runestone-docker)
   has instructions and everything you need to build the image,
   or you can just pull the latest from docker hub:


   ```
   docker pull dparillo/runestone
   ```

For details running and using the container, refer to the
[README](https://github.com/DaveParillo/runestone-docker/blob/master/README.md)

## Install and make a Python virtualenv
 
* Documentation here:  https://virtualenv.pypa.io/en/stable/
* Video here:  https://www.youtube.com/watch?v=IX-v6yvGYFg
* For the impatient:

```
    $ sudo pip install virtualenv
    $ virtualenv /path/to/some/directory
    $ source /path/to/some/directory/bin/activate
```
     
* You will need to do the last command **every time** you want to work on the book in your virtual environment.
If you have not used Python virtual environments before I strongly recommend reading the docs or watching the video
 
With the virtual environment installed and configured you can continue.

```
    $ pip install runestone

    $ runestone build -- will build the html and put it in ``./docs/``
    $ runestone serve   -- will start a webserver and serve the pages locally from ``./docs/``

```

