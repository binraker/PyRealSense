//run test file with execfile('2.py')

#include "include.h"

PXCSenseManager *sm;
PyArrayObject *frameIR;
PyArrayObject *frameRGB;
PyArrayObject *frameDepth;
static PyObject *RealSenseError;
pxcStatus sts;
PXCCapture::Sample *sample;
bool first = true;

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

	int type;
	if (format == PXCImage::PIXEL_FORMAT_Y8)
		type = NPY_UBYTE;
	else if (format == PXCImage::PIXEL_FORMAT_RGB24)
	{
		type = NPY_UBYTE;
		dims[2] = 3;
		shape = 3;
	}
	else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32)
		type = NPY_FLOAT32;

	PyArrayObject *npImage = (PyArrayObject *)PyArray_SimpleNewFromData(shape, dims, type, data.planes[0]);
	//PyArray_ENABLEFLAGS(npImage, NPY_ARRAY_OWNDATA);
	return npImage;

};

static PyObject* getDev(PyObject* self, PyObject* args)

{
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

}

static PyObject* relDev(PyObject* self, PyObject* args)
{
	sm->Release();
	return Py_BuildValue("i", 1);
}

/*static PyObject* getframes(PyObject* self, PyObject* args)
{
	//sm->FlushFrame();
	sts = sm->AcquireFrame();
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to aquire a a frame");
	sample = sm->QuerySample();
	//-
/*	if (!first)
	{
		PyArray_CLEARFLAGS(frameDepth, NPY_ARRAY_OWNDATA);
		first = false;
	}
	

	frameIR = PXC2numPy(sample->ir, PXCImage::PIXEL_FORMAT_Y8);
	frameRGB = PXC2numPy(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
	frameDepth = PXC2numPy(sample->depth, PXCImage::PIXEL_FORMAT_DEPTH_F32);
//	PyArray_ENABLEFLAGS(frameDepth, NPY_ARRAY_OWNDATA);
//	PyArray_ENABLEFLAGS(frameIR, NPY_ARRAY_OWNDATA);
//	PyArray_ENABLEFLAGS(frameRGB, NPY_ARRAY_OWNDATA);

	return Py_BuildValue("i", 1);
}
*/

static PyObject* getframes(PyObject* self, PyObject* args) {

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
	PyArray_ENABLEFLAGS(frameDepthl, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS(frameIRl, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS(frameRGBl, NPY_ARRAY_OWNDATA);
	PyObject *rslt = PyTuple_New(3);
	PyTuple_SetItem(rslt, 0, (PyObject*)frameDepthl);
	PyTuple_SetItem(rslt, 1, (PyObject*)frameIRl);
	PyTuple_SetItem(rslt, 2, (PyObject*)frameRGBl);
	return rslt;

}






static PyObject* relframes(PyObject* self, PyObject* args)
{
	sm->FlushFrame();
	//sample->ReleaseImages();
	sm->ReleaseFrame();
	return Py_BuildValue("i", 1);
}

static PyObject* getIR(PyObject* self, PyObject* args)
{
	//PyArray_ENABLEFLAGS(frameIR, NPY_ARRAY_OWNDATA);
	return PyArray_Return(frameIR);
}

static PyObject* getRGB(PyObject* self, PyObject* args)
{
	//PyArray_ENABLEFLAGS(frameRGB, NPY_ARRAY_OWNDATA);
	return PyArray_Return(frameRGB);
}

static PyObject* getDepth(PyObject* self, PyObject* args)
{
	PyArrayObject frame;
	frame = *frameDepth;
	PyArray_ENABLEFLAGS(&frame, NPY_ARRAY_OWNDATA);
	return PyArray_Return(&frame);	
	
	//return PyArray_Return(frameDepth);

}

static PyObject* OpenGetClose(PyObject* self, PyObject* args)
{

	int width = 640;
	int height = 480;
	float frameRate = 60;

	PyArrayObject *frameIR;
	PyArrayObject *frameRGB;
	PyArrayObject *frameDepth;

	//Initilize camera
	PXCSenseManager *sm = PXCSenseManager::CreateInstance();

	//enable streams
	sm->EnableStream(PXCCapture::STREAM_TYPE_IR, width, height, frameRate);
	sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, width, height, frameRate);
	sm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, width, height, frameRate);

	sm->Init();

	sm->AcquireFrame();
	PXCCapture::Sample *sample = sm->QuerySample();
	//-
	frameIR = PXC2numPy(sample->ir, PXCImage::PIXEL_FORMAT_Y8);
	frameRGB = PXC2numPy(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
	frameDepth = PXC2numPy(sample->depth, PXCImage::PIXEL_FORMAT_DEPTH_F32);

	sm->ReleaseFrame();

	sm->Release();

	return PyArray_Return(frameRGB);

}

PyMethodDef RealSenseMethods[] = 
{
	{ "getframe", (PyCFunction)getframes, METH_VARARGS },
	{ "relframe", (PyCFunction)relframes, METH_VARARGS },
	{ "getdev", (PyCFunction)getDev, METH_VARARGS },
	{ "reldev", (PyCFunction)relDev, METH_VARARGS },
	{ "getdepth", (PyCFunction)getDepth, METH_VARARGS },
	{ "getrgb", (PyCFunction)getRGB, METH_VARARGS },
	{ "getIR", (PyCFunction)getIR, METH_VARARGS },
	{0,0,0}
};

PyMODINIT_FUNC
initspam(void)
{
	PyObject *m;

	RealSenseError = PyErr_NewException("RealSenseError.error", NULL, NULL);
	Py_INCREF(RealSenseError);
	m = Py_InitModule("spam", RealSenseMethods);
	import_array();
	if (m == NULL)
		return;
	

}
