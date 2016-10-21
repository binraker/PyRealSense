#ifdef _DEBUG
#define _DEBUG_WAS_DEFINED 0
#undef _DEBUG
#endif

#include <Python.h>

#ifdef _DEBUG_WAS_DEFINED
#define _DEBUG
#endif // _DEBUG_WAS_DEFINED

#include <pxcsensemanager.h>
#include <numpy/arrayobject.h>

static char module_docstring[] =
"This module allows control of Intel RealSense cameras";



/*run test file with:
execfile('2.py')

*/
