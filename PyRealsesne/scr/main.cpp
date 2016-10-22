//run test file with execfile('2.py')

#include "include.h"
#include <stdlib.h>



PXCSenseManager *sm;
static PyObject *RealSenseError;
pxcStatus sts;


PyArrayObject *PXC2numPy(PXCImage *pxcImage, PXCImage::PixelFormat format)
{
	PXCImage::ImageData data;
	pxcImage->AcquireAccess(PXCImage::ACCESS_READ, format, &data);

	int width = pxcImage->QueryInfo().width;
	int height = pxcImage->QueryInfo().height;
	npy_intp dims[3] = { height, width,1 };
	int shape = 2;

	if (!format)
		format = pxcImage->QueryInfo().format;

	//bits commented out in these if statements tried to put data.planes[0] into a new array and return that. didn't even get first frame!!
	//PyArrayObject *npImage;
	int type;
	if (format == PXCImage::PIXEL_FORMAT_Y8)
	{
		type = NPY_UBYTE;
		/*unsigned char temp;
		temp = (unsigned char)malloc(sizeof(unsigned char)*height*width);
		temp = (unsigned char)data.planes[0];
		npImage = (PyArrayObject *)PyArray_SimpleNewFromData(shape, dims, type, &temp);
		//free(&temp);*/
	}
	else if (format == PXCImage::PIXEL_FORMAT_RGB24)
	{
		type = NPY_UBYTE;
		dims[2] = 3;
		shape = 3;
		/*unsigned char temp;
		temp = (unsigned char)malloc(sizeof(unsigned char)*height*width*3);
		temp = (unsigned char)data.planes[0];
		npImage = (PyArrayObject *)PyArray_SimpleNewFromData(shape, dims, type, &temp);
		//free(&temp);*/
	
	}
	else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32)
	{
		type = NPY_FLOAT32;

	/*	long temp;
		temp = (long)malloc(sizeof(long)*height*width);
		temp = (long)data.planes[0];
		npImage = (PyArrayObject *)PyArray_SimpleNewFromData(shape, dims, type, &temp);
		//free(&temp);*/
	}

	

	PyArrayObject *npImage = (PyArrayObject *)PyArray_SimpleNewFromData(shape, dims, type, data.planes[0]);
	
	
	//PyArray_ENABLEFLAGS(npImage, NPY_ARRAY_OWNDATA); //tried puttin it here, still didn't work
	pxcImage->ReleaseAccess(&data);
	
	return npImage;

};//helper function to convert PCX into numpy array

static PyObject* getDev(PyObject* self, PyObject* args)

{
	//TODO: let you choose camera settings and what frmaes you want to capture
	int width = 640;
	int height = 480;
	float frameRate = 60;

	//Initilize camera
	sm = PXCSenseManager::CreateInstance();

	//enable streams
	sm->EnableStream(PXCCapture::STREAM_TYPE_IR, width, height, frameRate);
	sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, width, height, frameRate);
	sm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, width, height, frameRate);

	sts = sm->Init();
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to initialize camera");
	return Py_BuildValue("i", sts);

}// get device

static PyObject* relDev(PyObject* self, PyObject* args)
{
	sm->Release();
	return Py_BuildValue("i", 1);
}//release device

static PyObject* getframes(PyObject* self, PyObject* args) {
	//TODO:add ability to select what frames you want

	PXCCapture::Sample *sample;
	PyArrayObject *frameIRl;
	PyArrayObject *frameRGBl;
	PyArrayObject *frameDepthl;

	sts = sm->AcquireFrame();
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to aquire a a frame");
	sample = sm->QuerySample();

	frameIRl = PXC2numPy(sample->ir, PXCImage::PIXEL_FORMAT_Y8);
	frameRGBl = PXC2numPy(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
	frameDepthl = PXC2numPy(sample->depth, PXCImage::PIXEL_FORMAT_DEPTH_F32);
// this bit here is the issue: 
	/*
	PyArray_ENABLEFLAGS(frameDepthl, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS(frameIRl, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS(frameRGBl, NPY_ARRAY_OWNDATA);*/
//one tuple of frames is retunred OK but the next request results in an access violation. eg:
//Exception thrown at 0x77887F1D (ntdll.dll) in python.exe: 0xC0000005: Access violation reading location 0x0C4F4E7A.
	PyObject *rslt = PyTuple_New(3);
	PyTuple_SetItem(rslt, 0, (PyObject*)frameDepthl);
	PyTuple_SetItem(rslt, 1, (PyObject*)frameIRl);
	PyTuple_SetItem(rslt, 2, (PyObject*)frameRGBl);

	sm->FlushFrame();
	sm->ReleaseFrame();
	return rslt;

}//get frame

//stuff to link functions to python.


PyMethodDef RealSenseMethods[] = 
{
	{ "getframe", (PyCFunction)getframes, METH_VARARGS },
	{ "getdev", (PyCFunction)getDev, METH_VARARGS },
	{ "reldev", (PyCFunction)relDev, METH_VARARGS },
	{0,0,0}
};

PyMODINIT_FUNC
initspam(void)
{
	PyObject *m;
	RealSenseError = PyErr_NewException("RealSenseError.error", NULL, NULL);
	Py_INCREF(RealSenseError);
	m = Py_InitModule("spam", RealSenseMethods, module_docstring);
	import_array();
	if (m == NULL)
		return;
}
