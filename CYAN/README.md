## Bitfusion Test Assignment

The project contains a simple C library (`lattice`, located in the `lib` directory) and a short test program (`latticexample`, located in the `bin` directory) that uses said library.
It uses CMake to build, but since it's so simple you can use any other build system (such as simple Makefiles). 

The functions in this library don't do much, and for your convenience they print out their names and most of their arguments. 

We want you to introduce an intercepting ("interposing") layer that can be placed between the test program and the library. 
This interposing layer would take the place of the native ("base") lattice library, do some highly advanced processing, pass the call to the base library, optionally do some more processing, then return a return value to the test program. 
All this has to be done in utmost isolation, without any changes to the test program, and without the test program ever realizing it is not talking to the base library directly. 

### Design Requirements

The solution needs to meet the following requirements:

1.  The code for the interposing layer has to be automatically generated. 
    We plan to add many more functions to the library in the future, and we do not want to manually write an interposing layer for each function. 
    You should write a code generator, in the language and format of your choice, that will output the code for intercepting each function. 
    The code generator tool can take the `lattice.h` header file as input, but not the implementation files (`lattice.c` and `main.c`).
    The automated tool can be *guided* - in addition to reading the header file, you can give it some manual description of the functions of their arguments,
    either directly in the `lattice.h` header file or in a separate configuration file. 
    Still, it is beneficial to keep this manual configuration to a minimum, in order to make supporting new functions easier and faster. 
    - You are free to write your own helper functions which are then called from the generated code, or use existing external libraries. 
      The guiding principle being the automation is that new functions can be supported without adding any new custom code, not that everything has to be generated.
    - Some help about the contents of the input and output files can be found in the Notes and Suggestions section. 

2.  The interposing itself has to be able to be swapped directly in place of the base `lattice` library. 
    You cannot make modifications to the test program `latticeexample`. 
    It is preferred (but not required) that you setup the project so that the base and interposed libraries can be swapped without recompiling the test program. 
    - This is done by creating a new library that has the exact same function names and signatures as the base library, but your own implementation of these functions. 
    - See the Notes and Suggestions section for information about how to achieve this on Linux and Windows. 

3.  The interposing layer has to be able to load the base library and call its functions. 
    It is not meant as a replacement, but just as a thin layer that sits between the program and the library. 
    - Since the interposed library has the same function names as the base library, it cannot be linked to the base library at compile time. 
      The base library and its functions have to be loaded at runtime. 
    - The `dlopen`/`dlsym` (Linux) and `LoadLibrary` (Windows) family of functions may be useful here. 
    
### Functionality

Now, to the actual functionality of the interposed layer. 
This is divided into three development stages

1.  **Intercept and Forward**: The first stage is to directly forward all arguments to the base library, and forward the return value back to the caller. 
    At this stage, you do not need any introspection in the arguments and their types, as you just forward them. 

2.  **Dump Info**: The second stage is to print out a detailed log of all called functions, their inputs, outputs, and return values, to a file. 
    Try to think of good ways to print different data types: integers, strings, pointers, buffers. 
    At this stage you need to know the argument types in order to be able to make useful logs, such as printing out string contents. 

3.  **Modify In-Place**: The third stage is to modify the data in-place, transparently to the test program and base library. 
    To verify that your interposed layer is working, scan all integer parameters, and double them if and only if they are odd numbers
    For this stage you need to know the argument types and also be able to read and write their values. 

    For **extra credit**, you may think of your own devious ways to manipulate the data, but here are some suggestions. 
    - (easy) Reverse all strings before calling the functions
    - (easy) Randomly return an error even though the function succeeded
    - (intermediate) XOR input and output buffers against some constant value
    - (intermediate) Double all strings (append another copy of the string at the end of each string)
    - (hard) Search all buffers for certain patterns (byte sequences that look look like as pointers, integers, or strings) and log them
    - (hard) Perform search-and-replace on strings

### Notes and Suggestions

The section contains some guidance to both explain the assignment itself and to help you with the parts that are not the main focus of the assignment. 

#### Code generation and function descriptions

In this assignment, we want you to create a code generator - a piece of software that will read the input files and from them generate the implementation code for a new interposed library. 

The main input file will be the `lattice.h` header file, which contains all the involved types, functions, and their signatures. 
The functions have been designed to be simple to parse, you can parse them manually, or use an established C language parser like `libclang`. 
You can assume any new functions that we add in the future will contain similarly simple argument types. 

But no matter how simple or advanced your parser is, there is some required information that cannot be obtain purely by looking at the function signatures. 
- Is this `const char*` used as an opaque pointer value, or is representing a string?
- Is this `size_t` representing a size of some buffer? If so, is it the number of bytes, the number of 4-byte integers, or something else?
- Which size parameter corresponds to which buffer? 
- Is this `int*` parameter pointing to a list of integers, an optional parameter, or an output parameter?

In our `lattice` library, these questions are answered in the documentation comments, but a code generation tool will not be able to meaningfully read those. 
Additionally, you may peek into the `lattice.c` implementation file for clarifications, but the code generator may not. 

To this end, you should create some machine-readable disambiguations. 
These can be included in the `lattice.h` header itself, but may also be in a separate configuration file. 
You can use any format you like for this - text, YAML, JSON, UML, code - but it should be easy to write for humans. 
Your code generator will then read this configuration in order to get the missing information about the functions and their arguments. 

Using the information gained from the header file and the configuration file, it will then be able to output the function implementation code. 

#### The generated code

The generated code for a single function should roughly look like this (in pseudocode):

```
// The function name and signature has to match those from lattice.h
int single_int(int a)
{
    // Do any manipulation of the input arguments here
    
    // Load the base library symbol of the same name, and cast it to a function pointer of the appropriate type
    // Replace get_base_library_symbol() with something that calls the platform-specific functions
    typedef (...) single_int_fn;
    single_int_fn base_function = get_base_library_symbol("single_int");
    
    // Call the base library function, passing all the arguments
    int ret = base_function(a);
    
    // Do any manipulation of the output arguments here
    
    // Return the return value
    return ret;
}
```

It is up to you to fill the missig parts - the function pointer type, the way to dynamically load the base function, and any manipulation of the arguments. 

#### Library interposing

In order for a library to successfully pose as another library, it has to contain the same function names and signatures. 

On Linux, there are then two methods of loading it at runtime:
- Use LD_PRELOAD to directly preload the library: `LD_PRELOAD=/path/to/interposed/liblatticeinterposed.so ./bin/latticeexample`
- Make it have the *exact same filename* as the base library ant set LD_LIBRARY_PATH to the containing folder:
  `LD_LIBRARY_PATH=/path/to/interposed/ ./bin/latticeexample`

If you cannot make it work at runtime, you can modify the project structure so that `latticexample` links to the interposed library instead of the base one. 
The important part is to not change the `latticexample` test program code, you can re-compile and re-link it. 

Note that even in this case, the interposed library still has to dynamically load the base library and its symbols at runtime. 

#### Dynamic loading of libraries and symbols

Because of how linking in C works, you cannot link the interposed library directly to the base library, because it has the same function names. 
Instead, in this assignment you are expected to load it dynamically at runtime. 
This step is generally platform-specific, you are free to use whatever platform you like, and for the test assignment we don't need cross-platform compatibility. 

On Linux, this is achieved using `dlopen()` and `dlsym()`. 
The `dlopen` function opens and loads the library itself. 
For this test assignment, you can safely use a hardcoded path to the library. 
The `dlsym` function loads a symbol (in our case it will be a function) from the library by name. 
On Windows, the equivalent functions are `LoadLibrary` and `GetProcAddress`, respectively. 

On either platform, both `dlsym()` and `GetProcAddress` will give you a pointer. 
In order to call this as a function, you will need to cast it to a function pointer of the appropriate type. 
Function pointer types in C are rather unintuitive, they can be made a little clearer with a `typedef`, but you may still want to consult some documentation online.
For example, for a function with a signature of `int add(int a, int b)`, the function pointer type and a way to call it is:

````
typedef int (*add_fn)(int, int);
add_fn base_add = get_base_library_symbol("add");
int ret = base_add(1, 2);
````

When calling dynamically loaded functions, it is very important that the number and types of arguments match those of the base library function. 
If they don't, you will not get any warnings, but the program may (or may not) crash. 
