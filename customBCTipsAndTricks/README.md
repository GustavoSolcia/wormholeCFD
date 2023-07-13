# OpenFoam custom boundary condition tutorial

## Configure user boundary condition library directories

First, create your boundary condition library (I'm assuming that you already started your OpenFOAM environment on terminal)


```
mkdir -p $WM_PROJECT_USER_DIR/src/finiteVolume/fields/fvPatchFields/derived
```

Then, you need a finiteVolume Make file

```
cd $WM_PROJECT_USER_DIR/src/finiteVolume

mkdir Make
```

Now we need a files (just substitute <yourBoundaryCondition> for your specific case):
  
Make sure that the third line of the code below is typed correctly according to you new boundary condition. Also, new boundary conditions should be placed bellow that line to also be compiled.

```
fvPatchFields = fields/fvPatchFields
derivedFvPatchFields = $(fvPatchFields)/derived
$(derivedFvPatchFields)/<yourBoundaryCondition>/<yourBoundaryCondition>FvPatchVectorField.C
LIB = $(FOAM_USER_LIBBIN)/libmyFiniteVolume
```

and options file:

```
EXE_INC = \
-I$(LIB_SRC)/finiteVolume/lnInclude
EXE_LIBS =
```

## Find a similar boundary condition and start modifications

The implementations can be found in:
```
$FOAM_SRC/finiteVolume/fields/fvPatchFields/  
```
  
## Compile library

```
wmake libso  
```
  
## Prepare case
