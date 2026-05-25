..  Copyright (C)  Dave Parillo.  Permission is granted to copy, distribute
    and/or modify this document under the terms of the GNU Free Documentation
    License, Version 1.3 or any later version published by the Free Software
    Foundation; with Invariant Sections being Forward, and Preface,
    no Front-Cover Texts, and no Back-Cover Texts.  A copy of
    the license is included in the section entitled "GNU Free Documentation
    License".

.. index:: pointers to pointers

Pointers to pointers
====================
A pointer can point to any memory address within the scope of the program,
which includes pointers themselves.
Each new pointer just adds another to the chain of pointers.
The language does not impose a strict limit.
The only limit is your sanity...

.. code-block:: cpp

   int x = 8;

   // all of these variables point to x
   int* p2x     = &x;
   int** p2p    = &p2x;
   int*** p2pp  = &p2p;

`See it for yourself. <http://pythontutor.com/cpp.html#code=%23include%20%3Ciostream%3E%0Aint%20main%28%29%20%7B%0A%20%20int%20x%20%3D%208%3B%0A%0A%20%20//%20all%20of%20these%20variables%20point%20to%20x%0A%20%20int*%20p2x%20%20%20%20%20%3D%20%26x%3B%0A%20%20int**%20p2p%20%20%20%20%3D%20%26p2x%3B%0A%20%20int***%20p2pp%20%20%3D%20%26p2p%3B%0A%0A%20%20return%200%3B%0A%7D&curInstr=3&mode=display&origin=opt-frontend.js&py=cpp&rawInputLstJSON=%5B%5D>`_

Like ``int`` or ``char``, a pointer type is still a type.
When you declare a variable of type pointer, 
storage still must be allocated somewhere,
and this storage must have an address too.

When dealing with pointers, we have to manage the added complexity
of keeping clear in our minds the difference between
*the pointer variable* and *what the pointer points to*.
When dealing with pointers to pointers, we have to manage
the pointer, what it points to, and *what the pointer that it points to points to*.

.. code-block:: cpp

   #include <iostream>
   #include <string>

   using std::string;
   using std::cout;

    int main() {
      string message[] = {"Alice","Bob here!","Carol checking in."};

      string *sp;   // a pointer to at least 1 string

      sp = message;
      cout << "sp:\n";
      cout << sp << '\n';
      cout << *sp << '\n';
      cout << *(sp + 1) << '\n';
      cout << *(sp + 2) << "\n\n";


      cout << "sp2:\n";
      string *sp2 = new string [3];          //create string pointer on the heap
      *sp2 = "\nAlice has left the building";
      *(sp2 + 1) = "Bob who?";
      *(sp2 + 2) = "Carol checked out.";

      cout << sp2 << '\n';
      cout << *sp2 << '\n';
      cout << *(sp2 + 1) << '\n';
      cout << *(sp2 + 2) << '\n' << '\n';

      string **sp3;                 // a pointer to a string pointer

      cout << "sp3:\n";
      sp3 = &sp2;
      cout << sp3 << '\n';
      cout << **sp3 << '\n';
    }

.. index::
   pair: video; pointer to reference

You can also define a pointer to a reference variable:

.. youtube:: 0QOxC7ADT80
   :http: https
   :divid: pointer_0QOxC7ADT80

-----

.. admonition:: More to Explore

   - `MyCodeSchool <http://www.mycodeschool.com>`__ video: 
     `Pointers in C/C++ playlist <https://www.youtube.com/playlist?list=PL2_aWCzGMAwLZp6LMUKI3cc7pgGsasm2_>`__ 


