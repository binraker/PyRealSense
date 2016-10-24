/*run test file with:
execfile('test.py')
*/
#include "include.h"

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
	int size = height * width;

	if (!format)
		format = pxcImage->QueryInfo().format;

	PyArrayObject *npImage;
	int type;
	if (format == PXCImage::PIXEL_FORMAT_Y8)
	{
		type = NPY_UBYTE;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}
	else if (format == PXCImage::PIXEL_FORMAT_RGB24)
	{
		type = NPY_UBYTE;
		dims[2] = 3;
		shape = 3;
		size *= 3;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}
	else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32)
	{
		type = NPY_FLOAT32;
		size *= sizeof(float);

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}
	memcpy(npImage->data, data.planes[0], size);
	PyArray_ENABLEFLAGS(npImage, NPY_OWNDATA);
	pxcImage->ReleaseAccess(&data);

	return npImage;
}//helper function to convert PCX into numpy array

static PyObject* getDev(PyObject* self, PyObject* args)

{
	//TODO: let you choose camera settings and what frames you want to capture
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
		PyErr_SetString(RealSenseError, "Failed to initialize depth camera");


//TODO: expose these settings to python.


	PXCCapture::Device *cp = sm->QueryCaptureManager()->QueryDevice();
/*
motion to range tradeoff
0 (short exposure, short range, and better motion) to 100 (long exposure and long range.)
*/
	sts = cp->SetIVCAMMotionRangeTradeOff(100);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set range of depth camera");
/*
projected pattern setting.
*/
	sts = cp->SetIVCAMAccuracy(PXCCapture::Device::IVCAM_ACCURACY_COARSE);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set accuracy of depth camera");
	
	/*
Filter
0 Skeleton: Reports the depth data for high fidelity (high confidence) pixels only, and all other pixels as invalid. For SR300, the depth range is up to 4m.
1 Raw: Raw depth image without any post-processing filters. For SR300, the depth range is up to 4m.
2 Raw + Gradients filter: Raw depth image  with the gradient filter applied. For SR300, the depth range is up to 4m.
3 Very close range :Very low smoothing effect with high sharpness, accuracy levels, and low noise artifacts. Good for any distances of up to 350mm for F200, and up to 2m for SR300.
4 Close range:Low smoothing effect with high sharpness and accuracy levels. The noise artifacts are optimized for distances between 350mm to 550mm for F200, and up to 2m for SR300.
5 Mid-range [Default] :Moderate smoothing effect optimized for distances between 550mm to 850mm for F200 and up to 2m for SR300 to balance between good sharpness level, high accuracy and moderate noise artifacts.
6 Far range :High smoothing effect for distances between 850mm to 1000mm for F200 and up to 4m for SR300 bringing good accuracy with moderate sharpness level.
7 Very far range :Very high smoothing effect to bring moderate accuracy level for distances above 1m for F200, and 0.8-2.5m (up to 4m) for SR300. Use together with the MotionRangeTradeOff property to increase the depth range.
*/
	sts = cp->SetIVCAMFilterOption(5);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set filter of depth camera");

	/*
	laser power 0 (lowest) to 16 (highest)
	*/
	sts = cp->SetIVCAMLaserPower(16);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set filter of depth camera");

	return Py_BuildValue("i", sts);


	


}// get device// get device

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

	PyArray_ENABLEFLAGS(frameDepthl, NPY_OWNDATA| NPY_ARRAY_ENSURECOPY);
	PyArray_ENABLEFLAGS(frameIRl, NPY_OWNDATA | NPY_ARRAY_ENSURECOPY);
	PyArray_ENABLEFLAGS(frameRGBl, NPY_OWNDATA | NPY_ARRAY_ENSURECOPY);

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
initPyRealSense(void)
{
	PyObject *m;
	RealSenseError = PyErr_NewException("RealSenseError.error", NULL, NULL);
	Py_INCREF(RealSenseError);
	m = Py_InitModule("PyRealSense", RealSenseMethods, module_docstring);
	import_array();
	if (m == NULL)
		return;
}