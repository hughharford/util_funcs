# **About Util_funcs**
# **Useful repo for collected all-sorts-of-things**
I've put this together so in case I don't have access to my machine,
I can quickly get hold of a huge range of massive resources.

No doubt there are considerably better versions out there, but this
is my one that I know well. Hence the coded filenames etc!

## **Useful sections**
### **cheatsheets**
 - see util_funcs/cheatsheets
A collection of useful ML, python and related cheatsheets
### **makefiles**
 - see util_funcs/makefiles - and the actual Makefile, of course
Collection of various makefiles
Lots of useful commands, though of course, set up required
### **test material, including bdd behaviour driven development**
these provides setup and examples for BDD, very handy
examples show simple classes and examples with and without parameter input

 - see
   - util_funcs/bdd_start (for the classes to be tested)
   - util_funcs/tests/features (for the BDD feature definitions)
   - util_funcs/tests/step_defs (for the test code to match features)

- see also REF: https://pytest-bdd.readthedocs.io/en/latest/index.html?highlight=run#organizing-your-scenarios


#### **run tests**
run, in the root folder of the repo:
  - make test
    - this will run pytest only, and provide coverage report
  - make test_bdd
    - this will run pytest and pytest_bdd tests, and provide coverage report
