#include "include.h"

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

	return npImage;

};


PXCSenseManager *pxcSeneManager;
PyArrayObject *frameIR;
PyArrayObject *frameRGB;
PyArrayObject *frameDepth;

static PyObject* getDev(PyObject* self, PyObject* args)

{
	int width = 640;
	int height = 480;
	float frameRate = 60;

	//Initilize camera
	pxcSeneManager = PXCSenseManager::CreateInstance();

	//enable streams
	pxcSeneManager->EnableStream(PXCCapture::STREAM_TYPE_IR, width, height, frameRate);
	pxcSeneManager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, width, height, frameRate);
	pxcSeneManager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, width, height, frameRate);

	pxcSeneManager->Init();
	return Py_BuildValue("i", 1);
}

static PyObject* relDev(PyObject* self, PyObject* args)
{
	pxcSeneManager->Release();
	return Py_BuildValue("i", 1);
}

static PyObject* getframes(PyObject* self, PyObject* args)
{
	//PyArrayObject *frameIR;
	//PyArrayObject *frameRGB;
	//PyArrayObject *frameDepth;
	


	pxcSeneManager->AcquireFrame();
	PXCCapture::Sample *sample = pxcSeneManager->QuerySample();
	//-
	frameIR = PXC2numPy(sample->ir, PXCImage::PIXEL_FORMAT_Y8);
	frameRGB = PXC2numPy(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
	frameDepth = PXC2numPy(sample->depth, PXCImage::PIXEL_FORMAT_DEPTH_F32);
	
	
	return Py_BuildValue("i", 1);
	
}

static PyObject* relframes(PyObject* self, PyObject* args)
{
	pxcSeneManager->ReleaseFrame();
	return Py_BuildValue("i", 1);
}

static PyObject* getIR(PyObject* self, PyObject* args)
{
	return PyArray_Return(frameIR);
}

static PyObject* getRGB(PyObject* self, PyObject* args)
{
	return PyArray_Return(frameRGB);
}

static PyObject* getDepth(PyObject* self, PyObject* args)
{
	return PyArray_Return(frameDepth);
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
	PXCSenseManager *pxcSeneManager = PXCSenseManager::CreateInstance();

	//enable streams
	pxcSeneManager->EnableStream(PXCCapture::STREAM_TYPE_IR, width, height, frameRate);
	pxcSeneManager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, width, height, frameRate);
	pxcSeneManager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, width, height, frameRate);

	pxcSeneManager->Init();

	pxcSeneManager->AcquireFrame();
	PXCCapture::Sample *sample = pxcSeneManager->QuerySample();
	//-
	frameIR = PXC2numPy(sample->ir, PXCImage::PIXEL_FORMAT_Y8);
	frameRGB = PXC2numPy(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
	frameDepth = PXC2numPy(sample->depth, PXCImage::PIXEL_FORMAT_DEPTH_F32);

	pxcSeneManager->ReleaseFrame();

	pxcSeneManager->Release();

	return PyArray_Return(frameRGB);

}

PyMethodDef SpamMethods[] = 
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

	m = Py_InitModule("spam", SpamMethods);
	import_array();
	if (m == NULL)
		return;

}
