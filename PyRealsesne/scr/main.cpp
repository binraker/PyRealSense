#include "include.h"

PXCSenseManager *sm;
PXCCapture::Device *cp;
static PyObject *RealSenseError;
pxcStatus sts;
PXCSession *session;
bool streams[] = { false, false, false }; //depth,IR,colour

//helper functions(proabbly should be more of theses and less copy/past stuff!)
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

	else if (format == PXCImage::PIXEL_FORMAT_Y16)
	{
		type = NPY_USHORT;
		size *= 2;

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
	else if (format == PXCImage::PIXEL_FORMAT_RGB32)
	{
		type = NPY_UBYTE;
		dims[2] = 3;
		shape = 3;
		size *= 4;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}
	else if (format == PXCImage::PIXEL_FORMAT_YUY2)
	{
		type = NPY_UBYTE;
		size *= 2;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}

	else if (format == PXCImage::PIXEL_FORMAT_DEPTH)
	{
		type = NPY_USHORT;
		size *= 2;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}

	else if (format == PXCImage::PIXEL_FORMAT_DEPTH_RAW)
	{
		type = NPY_USHORT;
		size *= 2;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}
	else if (format == PXCImage::PIXEL_FORMAT_DEPTH_CONFIDENCE)
	{
		type = NPY_UBYTE;

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}

	else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32)
	{
		type = NPY_FLOAT32;
		size *= sizeof(float);

		npImage = (PyArrayObject *)PyArray_SimpleNew(shape, dims, type);
	}

//ToDo: support for N12 and possibly IR relative if i can be bothered

	memcpy(npImage->data, data.planes[0], size);
	PyArray_ENABLEFLAGS(npImage, NPY_OWNDATA);
	pxcImage->ReleaseAccess(&data);

	return npImage;
}//helper function to convert PCX into numpy array

PyObject* PropertyInfo2numPy(PXCCapture::Device::PropertyInfo input)
{
	PyObject *range = PyList_New(2);
	PyList_SetItem(range, 0, (PyObject*)Py_BuildValue("f", input.range.min));
	PyList_SetItem(range, 1, (PyObject*)Py_BuildValue("f", input.range.max));
	PyObject *ret = PyList_New(4);
	PyList_SetItem(ret, 0, (PyObject*)range);
	PyList_SetItem(ret, 1, (PyObject*)Py_BuildValue("f", input.step));
	PyList_SetItem(ret, 2, (PyObject*)Py_BuildValue("f", input.defaultValue));
	PyList_SetItem(ret, 3, (PyObject*)Py_BuildValue("O", input.automatic ? Py_True : Py_False));
	return ret;
}

//main control functions
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
		PyErr_SetString(RealSenseError, "Failed to initialize cameras");

	//get device so that functions that alter harware settings can access them
	cp = sm->QueryCaptureManager()->QueryDevice();
	//fet a sessioon handeler
	session = sm->QuerySession();

	return Py_BuildValue("i", sts);

} // get device

static PyObject* relDev(PyObject* self, PyObject* args)
{
	//cp->Release();
	sm->Release();
	return Py_BuildValue("i", 1);
} //release device

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

	PyArray_ENABLEFLAGS(frameDepthl, NPY_OWNDATA | NPY_ARRAY_ENSURECOPY);
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

//functions for general module

static PyObject* getCal(PyObject* self, PyObject* args)
{
	PXCProjection *projection = cp->CreateProjection();
	PXCCalibration *calib = projection->QueryInstance<PXCCalibration>();

	int cam;
	PXCCapture::StreamType streamtype = PXCCapture::STREAM_TYPE_IR;
	PXCCalibration::StreamCalibration cal;
	PXCCalibration::StreamTransform trans;

	if (!PyArg_ParseTuple(args, "i", &cam))
		return NULL;

	if (cam == 1)
		streamtype = PXCCapture::STREAM_TYPE_IR;
	else if (cam == 0)
		streamtype = PXCCapture::STREAM_TYPE_DEPTH;
	else if (cam == 2)
		streamtype = PXCCapture::STREAM_TYPE_COLOR;
	else
		PyErr_SetString(RealSenseError, "no such camera to get cal settings for");

	sts = calib->QueryStreamProjectionParameters(streamtype, &cal, &trans);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to get cal settings for camera");

	PyObject *focalLength = PyList_New(2);
	PyList_SetItem(focalLength, 0 ,(PyObject*)Py_BuildValue("f", cal.focalLength.x));
	PyList_SetItem(focalLength, 1, (PyObject*)Py_BuildValue("f", cal.focalLength.y));

	PyObject *principalPoint = PyList_New(2);
	PyList_SetItem(principalPoint, 0 ,(PyObject*)Py_BuildValue("f", cal.principalPoint.x));
	PyList_SetItem(principalPoint, 1, (PyObject*)Py_BuildValue("f", cal.principalPoint.y));
	
	PyObject *radialDistortion = PyList_New(3);
	PyList_SetItem(radialDistortion, 0, (PyObject*)Py_BuildValue("f", cal.radialDistortion[0]));
	PyList_SetItem(radialDistortion, 1, (PyObject*)Py_BuildValue("f", cal.radialDistortion[1]));
	PyList_SetItem(radialDistortion, 2, (PyObject*)Py_BuildValue("f", cal.radialDistortion[2]));

	PyObject *tangentialDistortion = PyList_New(2);
	PyList_SetItem(tangentialDistortion, 0, (PyObject*)Py_BuildValue("f", cal.tangentialDistortion[0]));
	PyList_SetItem(tangentialDistortion, 1, (PyObject*)Py_BuildValue("f", cal.tangentialDistortion[1]));
	PyObject *ret = PyList_New(4);
	
	PyList_SetItem(ret, 0, focalLength);
	PyList_SetItem(ret, 1, principalPoint);
	PyList_SetItem(ret, 2, radialDistortion);
	PyList_SetItem(ret, 3, tangentialDistortion);

	return ret;
}

static PyObject* getCoordinateSystem(PyObject* self, PyObject* args)
{
	PXCSession::CoordinateSystem system = session->QueryCoordinateSystem();
	if (system == PXCSession::COORDINATE_SYSTEM_FRONT_DEFAULT)
		return Py_BuildValue("i", 0);
	else if (system == PXCSession::COORDINATE_SYSTEM_REAR_DEFAULT)
		return Py_BuildValue("i", 0);
	else if (system == PXCSession::COORDINATE_SYSTEM_REAR_OPENCV)
		return Py_BuildValue("i", 0);

}

static PyObject* setCoordinateSystem(PyObject* self, PyObject* args)
{
	int system;

	if (!PyArg_ParseTuple(args, "i", &system))
		return NULL;

	if (system == 0)
		sts = session->SetCoordinateSystem(PXCSession::COORDINATE_SYSTEM_FRONT_DEFAULT);

	else if (system == 1)
		sts = session->SetCoordinateSystem(PXCSession::COORDINATE_SYSTEM_REAR_DEFAULT);

	else if (system == 2)
		sts = session->SetCoordinateSystem(PXCSession::COORDINATE_SYSTEM_REAR_OPENCV);

	if (sts != PXC_STATUS_NO_ERROR)
			PyErr_SetString(RealSenseError, "Failed to set coordinate system");
	return Py_BuildValue("i", 1);

}

static PyObject* getSDKAPI(PyObject* self, PyObject* args)
{
	PXCSession::ImplVersion ver;
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, (PyObject*)ver.major);
	PyTuple_SetItem(rslt, 1, (PyObject*)ver.minor);
	return rslt;
}

//F200 and SR300 functions

static PyObject* getRange(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryIVCAMMotionRangeTradeOff());
}

static PyObject* getAccuracy(PyObject* self, PyObject* args)
{
	PXCCapture::Device::IVCAMAccuracy res = cp->QueryIVCAMAccuracy();
	if (res == PXCCapture::Device::IVCAM_ACCURACY_COARSE)
		return Py_BuildValue("i", 0);
	else if (res == PXCCapture::Device::IVCAM_ACCURACY_MEDIAN)
		return Py_BuildValue("i", 1);
	else if (res == PXCCapture::Device::IVCAM_ACCURACY_FINEST)
		return Py_BuildValue("i", 2);
	else
		PyErr_SetString(RealSenseError, "camera returned unsuported accuracy");
}

static PyObject* getFilter(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryIVCAMFilterOption());
}

static PyObject* getLaserPower(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryIVCAMLaserPower());
}

static PyObject* setRange(PyObject* self, PyObject* args)
{
	int Range;

	if (!PyArg_ParseTuple(args, "i", &Range))
		return NULL;
	if (Range < 0)
		Range = 0;
	if (Range > 100)
		Range = 100;
	/*motion to range tradeoff
		0 (short exposure, short range, and better motion) to 100 (long exposure and long range.)
		*/
	sts = cp->SetIVCAMMotionRangeTradeOff(Range);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set range of depth camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setAccuracy(PyObject* self, PyObject* args)
{
	int Accuracy;

	if (!PyArg_ParseTuple(args, "i", &Accuracy))
		return NULL;
	if (Accuracy < 0)
		Accuracy = 0;
	if (Accuracy > 2)
		Accuracy = 2;
	/*
	projected pattern setting.
	*/
	switch (Accuracy)
	{
	case 0:
		sts = cp->SetIVCAMAccuracy(PXCCapture::Device::IVCAM_ACCURACY_COARSE);
		if (sts != PXC_STATUS_NO_ERROR)
			PyErr_SetString(RealSenseError, "Failed to set accuracy of depth camera");
	case 1:
		sts = cp->SetIVCAMAccuracy(PXCCapture::Device::IVCAM_ACCURACY_MEDIAN);
		if (sts != PXC_STATUS_NO_ERROR)
			PyErr_SetString(RealSenseError, "Failed to set accuracy of depth camera");
	case 2:
		sts = cp->SetIVCAMAccuracy(PXCCapture::Device::IVCAM_ACCURACY_FINEST);
		if (sts != PXC_STATUS_NO_ERROR)
			PyErr_SetString(RealSenseError, "Failed to set accuracy of depth camera");
	default:
		break;
	}
	return Py_BuildValue("i", 1);
}

static PyObject* setFilter(PyObject* self, PyObject* args)
{
	int filter;

	if (!PyArg_ParseTuple(args, "i", &filter))
		return NULL;
	if (filter < 0)
		filter = 0;
	if (filter > 7)
		filter = 7;
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
	sts = cp->SetIVCAMFilterOption(filter);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set filter of depth camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setLaserPower(PyObject* self, PyObject* args)
{
	/*
	laser power 0 (lowest) to 16 (highest)
	*/

	int power;

	if (!PyArg_ParseTuple(args, "i", &power))
		return NULL;
	if (power < 0)
		power = 0;
	if (power > 16)
		power = 16;

	sts = cp->SetIVCAMLaserPower(power);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set filter of depth camera");
	return Py_BuildValue("i", 1);
}

static PyObject* getAccuracyDefault(PyObject* self, PyObject* args)
{
	PXCCapture::Device::IVCAMAccuracy res = cp->QueryIVCAMAccuracyDefaultValue();
	if (res == PXCCapture::Device::IVCAM_ACCURACY_COARSE)
		return Py_BuildValue("i", 0);
	else if (res == PXCCapture::Device::IVCAM_ACCURACY_MEDIAN)
		return Py_BuildValue("i", 1);
	else if (res == PXCCapture::Device::IVCAM_ACCURACY_FINEST)
		return Py_BuildValue("i", 2);
	else
		PyErr_SetString(RealSenseError, "camera returned unsuported accuracy");
	return Py_BuildValue("i", res);
}

static PyObject* getFilterInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryIVCAMFilterOptionInfo());
}

static PyObject* getLaserPowerInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryIVCAMLaserPowerInfo());
}

static PyObject* getRangeInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryIVCAMMotionRangeTradeOffInfo());
}


//General functions
static PyObject* getColorAutoExposure(PyObject* self, PyObject* args)
{
	return Py_BuildValue("O", cp->QueryColorAutoExposure() ? Py_True : Py_False);
}

static PyObject* getColorAutoPowerLineFrequency(PyObject* self, PyObject* args)
{
	return Py_BuildValue("O", cp->QueryColorAutoPowerLineFrequency() ? Py_True : Py_False);
}

static PyObject* getColorAutoWhiteBalance(PyObject* self, PyObject* args)
{
	return Py_BuildValue("O", cp->QueryColorAutoWhiteBalance() ? Py_True : Py_False);
}

static PyObject* getColorBackLightCompensation(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorBackLightCompensation());
}

static PyObject* getColorBackLightCompensationInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorBackLightCompensationInfo());
}

static PyObject* getColorBrightness(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorBrightness());
}

static PyObject* getColorBrightnessInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorBrightnessInfo());
}

static PyObject* getColorContrast(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorContrast());
}

static PyObject* getColorContrastInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorContrastInfo());
}

static PyObject* getColorExposure(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorExposure());
}

static PyObject* getColorExposureInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorExposureInfo());
}

static PyObject* getColorHue(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorHue());
}

static PyObject* getColorHueInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorHueInfo());
}

static PyObject* getColorFieldOfView(PyObject* self, PyObject* args)
{
	PXCPointF32 val = cp->QueryColorFieldOfView();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("f", val.x));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("f", val.y));
	return rslt;
}

static PyObject* getColorFocalLength(PyObject* self, PyObject* args)
{
	PXCPointF32 val = cp->QueryColorFocalLength();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("f", val.x));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("f", val.y));
	return rslt;
}

static PyObject* getColorFocalLengthMM(PyObject* self, PyObject* args)
{
	return Py_BuildValue("f", cp->QueryColorFocalLengthMM());
}

static PyObject* getColorGain(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorGain());
}

static PyObject* getColorGainInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorGainInfo());
}

static PyObject* getColorGamma(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorGamma());
}

static PyObject* getColorGammaInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorGammaInfo());
}

static PyObject* getColorPowerLineFrequency(PyObject* self, PyObject* args)
{
	PXCCapture::Device::PowerLineFrequency val = cp->QueryColorPowerLineFrequency();
	if (val == PXCCapture::Device::POWER_LINE_FREQUENCY_50HZ)
		return Py_BuildValue("i", 50);
	if (val == PXCCapture::Device::POWER_LINE_FREQUENCY_60HZ)
		return Py_BuildValue("i", 60);
	if (val == PXCCapture::Device::POWER_LINE_FREQUENCY_DISABLED)
		return Py_BuildValue("i", 0);
}

static PyObject* getColorPrincipalPoint(PyObject* self, PyObject* args)
{
	PXCPointF32 val = cp->QueryColorPrincipalPoint();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("i", val.x));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("i", val.y));
	return rslt;
}

static PyObject* getColorSaturation(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorSaturation());
}

static PyObject* getColorSaturationInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorSaturationInfo());
}

static PyObject* getColorSharpness(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorSharpness());
}

static PyObject* getColorSharpnessInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorSharpnessInfo());
}

static PyObject* getColorWhiteBalance(PyObject* self, PyObject* args)
{
	return Py_BuildValue("i", cp->QueryColorWhiteBalance());
}

static PyObject* getColorWhiteBalanceInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryColorWhiteBalanceInfo());
}

static PyObject* getDepthConfidenceThreshold(PyObject* self, PyObject* args)
{
	return Py_BuildValue("h", cp->QueryDepthConfidenceThreshold());
}

static PyObject* getDepthConfidenceThresholdInfo(PyObject* self, PyObject* args)
{
	return PropertyInfo2numPy(cp->QueryDepthConfidenceThresholdInfo());
}

static PyObject* getDepthFieldOfView(PyObject* self, PyObject* args)
{
	PXCPointF32 val = cp->QueryDepthFieldOfView();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("f", val.x));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("f", val.y));
	return rslt;
}

static PyObject* getDepthFocalLength(PyObject* self, PyObject* args)
{
	PXCPointF32 val = cp->QueryDepthFocalLength();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("f", val.x));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("f", val.y));
	return rslt;
}

static PyObject* getDepthFocalLengthMM(PyObject* self, PyObject* args)
{
	return Py_BuildValue("f", cp->QueryDepthFocalLengthMM());
}

static PyObject* getDepthLowConfidenceValue(PyObject* self, PyObject* args)
{
	return Py_BuildValue("H", cp->QueryDepthLowConfidenceValue());
}

static PyObject* getDepthPrincipalPoint(PyObject* self, PyObject* args)
{
	PXCPointF32 val = cp->QueryDepthPrincipalPoint();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("f", val.x));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("f", val.y));
	return rslt;
}

static PyObject* getDepthSensorRange(PyObject* self, PyObject* args)
{
	PXCRangeF32 val = cp->QueryDepthSensorRange();
	PyObject *rslt = PyTuple_New(2);
	PyTuple_SetItem(rslt, 0, Py_BuildValue("f", val.min));
	PyTuple_SetItem(rslt, 1, Py_BuildValue("f", val.max));
	return rslt;
}

static PyObject* getDeviceAllowProfileChange(PyObject* self, PyObject* args)
{
	return Py_BuildValue("O", cp->QueryDeviceAllowProfileChange() ? Py_True : Py_False);
}

static PyObject* getDepthUnit(PyObject* self, PyObject* args)
{
	return Py_BuildValue("f", cp->QueryDepthUnit());
}

static PyObject* getMirrorMode(PyObject* self, PyObject* args)
{
	PXCCapture::Device::MirrorMode val = cp->QueryMirrorMode();
	if (val == PXCCapture::Device::MIRROR_MODE_DISABLED)
		return Py_BuildValue("O", Py_False);
	if (val == PXCCapture::Device::MIRROR_MODE_HORIZONTAL)
		return Py_BuildValue("O", Py_True);
}

static PyObject* setColorAutoExposure(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;

	sts = cp->SetColorAutoExposure(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set Auto exposure of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorAutoWhiteBalance(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;

	sts = cp->SetColorAutoWhiteBalance(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set white ballance of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorAutoPowerLineFrequency(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;

	sts = cp->SetColorAutoPowerLineFrequency(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set power line freq of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorBackLightCompensation(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	PXCCapture::Device::PropertyInfo info = cp->QueryColorBackLightCompensationInfo();
	if (val < info.range.min)
		val = info.range.min;
	if (val > info.range.max)
		val = info.range.max;

	sts = cp->SetColorBackLightCompensation(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set backlight compensation of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorBrightness(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val < -10000)
		val = -10000;
	if (val > 10000)
		val = 10000;

	sts = cp->SetColorBrightness(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set brightness of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorContrast(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val < 0)
		val = 0;
	if (val > 10000)
		val = 10000;

	sts = cp->SetColorContrast(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set contrast of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorExposure(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	PXCCapture::Device::PropertyInfo info = cp->QueryColorExposureInfo();
	if (val < info.range.min)
		val = info.range.min;
	if (val > info.range.max)
		val = info.range.max;

	sts = cp->SetColorExposure(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set exposure of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorHue(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val < -18000)
		val = -18000;
	if (val > 18000)
		val = 18000;

	sts = cp->SetColorHue(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set hue of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorGamma(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val < 0)
		val = 0;
	if (val > 500)
		val = 500;

	sts = cp->SetColorGamma(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set gamma of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorGain(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	PXCCapture::Device::PropertyInfo info = cp->QueryColorGainInfo();
	if (val < info.range.min)
		val = info.range.min;
	if (val > info.range.max)
		val = info.range.max;

	sts = cp->SetColorGain(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set gain of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorPowerLineFrequency(PyObject* self, PyObject* args)
{
	int val;
	PXCCapture::Device::PowerLineFrequency opt = PXCCapture::Device::POWER_LINE_FREQUENCY_DISABLED;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val == 50)
		opt = PXCCapture::Device::POWER_LINE_FREQUENCY_50HZ;
	else if (val == 60)
		opt = PXCCapture::Device::POWER_LINE_FREQUENCY_60HZ;
	else
		opt = PXCCapture::Device::POWER_LINE_FREQUENCY_DISABLED;

	sts = cp->SetColorPowerLineFrequency(opt);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set power line frequency of colour camera");

	return Py_BuildValue("i", 1);
}

static PyObject* setColorSaturation(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val < 0)
		val = 0;
	if (val > 10000)
		val = 10000;
	sts = cp->SetColorSaturation(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set saturation of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorSharpness(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (val < 0)
		val = 0;
	if (val > 100)
		val = 100;
	sts = cp->SetColorSharpness(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set sharpness of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setColorWhiteBalance(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	PXCCapture::Device::PropertyInfo info = cp->QueryColorWhiteBalanceInfo();
	if (val < info.range.min)
		val = info.range.min;
	if (val > info.range.max)
		val = info.range.max;

	sts = cp->SetColorWhiteBalance(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set white ballance of colour camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setDepthConfidenceThreshold(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "H", &val))
		return NULL;
	PXCCapture::Device::PropertyInfo info = cp->QueryDepthConfidenceThresholdInfo();
	if (val < info.range.min)
		val = info.range.min;
	if (val > info.range.max)
		val = info.range.max;

	sts = cp->SetDepthConfidenceThreshold(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set confidence threashold of depth camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setDepthUnit(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "f", &val))
		return NULL;

	sts = cp->SetDepthUnit(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set depth unit of depth camera");
	return Py_BuildValue("i", 1);
}

static PyObject* setDeviceAllowProfileChange(PyObject* self, PyObject* args)
{
	bool val;

	if (!PyArg_ParseTuple(args, "p", &val))
		return NULL;

	sts = cp->SetDeviceAllowProfileChange(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set allow profile change");
	return Py_BuildValue("i", 1);
}

static PyObject* setMirrorMode(PyObject* self, PyObject* args)
{
	int val;
	PXCCapture::Device::MirrorMode opt;

	if (!PyArg_ParseTuple(args, "i", &val))
		return NULL;
	if (!val)
		opt = PXCCapture::Device::MIRROR_MODE_DISABLED;
	else
		opt = PXCCapture::Device::MIRROR_MODE_HORIZONTAL;

	sts = cp->SetMirrorMode(opt);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set mirror mode");

	return Py_BuildValue("i", 1);
}
/*



static PyObject* set(PyObject* self, PyObject* args)
{
	int val;

	if (!PyArg_ParseTuple(args, "", &val))
		return NULL;
	if (val < 0)
		val = 0;
	if (val > 16)
		val = 16;

	sts = cp->Set(val);
	if (sts != PXC_STATUS_NO_ERROR)
		PyErr_SetString(RealSenseError, "Failed to set  of  camera");
	return Py_BuildValue("i", 1);
}



static PyObject* get(PyObject* self, PyObject* args)
{
	return Py_BuildValue("", cp->Query());
}


*/



//stuff to link functions to python.

PyMethodDef RealSenseMethods[] =
{
	{ "getframe", (PyCFunction)getframes, METH_VARARGS },
	{ "getdev", (PyCFunction)getDev, METH_VARARGS },
	{ "reldev", (PyCFunction)relDev, METH_VARARGS },
	{ "setDepthLaserPower", (PyCFunction)setLaserPower, METH_VARARGS },
	{ "setDepthFilter", (PyCFunction)setFilter, METH_VARARGS },
	{ "setDepthAccuracy", (PyCFunction)setAccuracy, METH_VARARGS },
	{ "setDepthRange", (PyCFunction)setRange, METH_VARARGS },
	{ "getDepthLaserPower", (PyCFunction)getLaserPower, METH_VARARGS },
	{ "getDepthFilter", (PyCFunction)getFilter, METH_VARARGS },
	{ "getDepthAccuracy", (PyCFunction)getAccuracy, METH_VARARGS },
	{ "getDepthRange", (PyCFunction)getRange, METH_VARARGS },
	{ "getColorAutoExposure", (PyCFunction)getColorAutoExposure, METH_VARARGS },
	{ "getColorAutoPowerLineFrequency", (PyCFunction)getColorAutoPowerLineFrequency, METH_VARARGS },
	{ "getColorAutoWhiteBalance", (PyCFunction)getColorAutoWhiteBalance, METH_VARARGS },
	{ "getColorBackLightCompensation", (PyCFunction)getColorBackLightCompensation, METH_VARARGS },
	{ "getColorBackLightCompensationInfo", (PyCFunction)getColorBackLightCompensationInfo, METH_VARARGS },
	{ "getColorBrightness", (PyCFunction)getColorBrightness, METH_VARARGS },
	{ "getColorBrightnessInfo", (PyCFunction)getColorBrightnessInfo, METH_VARARGS },
	{ "getColorContrast", (PyCFunction)getColorContrast, METH_VARARGS },
	{ "getColorContrastInfo", (PyCFunction)getColorContrastInfo, METH_VARARGS },
	{ "getColorExposure", (PyCFunction)getColorExposure, METH_VARARGS },
	{ "getColorExposureInfo", (PyCFunction)getColorExposureInfo, METH_VARARGS },
	{ "getColorHue", (PyCFunction)getColorHue, METH_VARARGS },
	{ "getColorHueInfo", (PyCFunction)getColorHueInfo, METH_VARARGS },
	{ "getColorFieldOfView", (PyCFunction)getColorFieldOfView, METH_VARARGS },
	{ "getColorFocalLength", (PyCFunction)getColorFocalLength, METH_VARARGS },
	{ "getColorFocalLengthMM", (PyCFunction)getColorFocalLengthMM, METH_VARARGS },
	{ "getColorGain", (PyCFunction)getColorGain, METH_VARARGS },
	{ "getColorGainInfo", (PyCFunction)getColorGainInfo, METH_VARARGS },
	{ "getColorGamma", (PyCFunction)getColorGamma, METH_VARARGS },
	{ "getColorGammaInfo", (PyCFunction)getColorGammaInfo, METH_VARARGS },
	{ "getColorPowerLineFrequency", (PyCFunction)getColorPowerLineFrequency, METH_VARARGS },
	{ "getColorPrincipalPoint", (PyCFunction)getColorPrincipalPoint, METH_VARARGS },
	{ "getColorSaturation", (PyCFunction)getColorSaturation, METH_VARARGS },
	{ "getColorSaturationInfo", (PyCFunction)getColorSaturationInfo, METH_VARARGS },
	{ "getColorSharpness", (PyCFunction)getColorSharpness, METH_VARARGS },
	{ "getColorSharpnessInfo", (PyCFunction)getColorSharpnessInfo, METH_VARARGS },
	{ "getColorWhiteBalance", (PyCFunction)getColorWhiteBalance, METH_VARARGS },
	{ "getColorWhiteBalanceInfo", (PyCFunction)getColorWhiteBalanceInfo, METH_VARARGS },
	{ "getDepthConfidenceThreshold", (PyCFunction)getDepthConfidenceThreshold, METH_VARARGS },
	{ "getDepthConfidenceThresholdInfo", (PyCFunction)getDepthConfidenceThresholdInfo, METH_VARARGS },
	{ "getDepthFieldOfView", (PyCFunction)getDepthFieldOfView, METH_VARARGS },
	{ "getDepthFocalLength", (PyCFunction)getDepthFocalLength, METH_VARARGS },
	{ "getDepthFocalLengthMM", (PyCFunction)getDepthFocalLengthMM, METH_VARARGS },
	{ "getDepthLowConfidenceValue", (PyCFunction)getDepthLowConfidenceValue, METH_VARARGS },
	{ "getDepthPrincipalPoint", (PyCFunction)getDepthPrincipalPoint, METH_VARARGS },
	{ "getDepthSensorRange", (PyCFunction)getDepthSensorRange, METH_VARARGS },
	{ "getDeviceAllowProfileChange", (PyCFunction)getDeviceAllowProfileChange, METH_VARARGS },
	{ "getDepthUnit", (PyCFunction)getDepthUnit, METH_VARARGS },
	{ "getMirrorMode", (PyCFunction)getMirrorMode, METH_VARARGS },
	{ "setColorAutoExposure", (PyCFunction)setColorAutoExposure, METH_VARARGS },
	{ "setColorAutoWhiteBalance", (PyCFunction)setColorAutoWhiteBalance, METH_VARARGS },
	{ "setColorAutoPowerLineFrequency", (PyCFunction)setColorAutoPowerLineFrequency, METH_VARARGS },
	{ "setColorBackLightCompensation", (PyCFunction)setColorBackLightCompensation, METH_VARARGS },
	{ "setColorBrightness", (PyCFunction)setColorBrightness, METH_VARARGS },
	{ "setColorContrast", (PyCFunction)setColorContrast, METH_VARARGS },
	{ "setColorExposure", (PyCFunction)setColorExposure, METH_VARARGS },
	{ "setColorHue", (PyCFunction)setColorHue, METH_VARARGS },
	{ "setColorGamma", (PyCFunction)setColorGamma, METH_VARARGS },
	{ "setColorGain", (PyCFunction)setColorGain, METH_VARARGS },
	{ "setColorPowerLineFrequency", (PyCFunction)setColorPowerLineFrequency, METH_VARARGS },
	{ "setColorSaturation", (PyCFunction)setColorSaturation, METH_VARARGS },
	{ "setColorSharpness", (PyCFunction)setColorSharpness, METH_VARARGS },
	{ "setColorWhiteBalance", (PyCFunction)setColorWhiteBalance, METH_VARARGS },
	{ "setDepthConfidenceThreshold", (PyCFunction)setDepthConfidenceThreshold, METH_VARARGS },
	{ "setDepthUnit", (PyCFunction)setDepthUnit, METH_VARARGS },
	{ "setDeviceAllowProfileChange", (PyCFunction)setDeviceAllowProfileChange, METH_VARARGS },
	{ "setMirrorMode", (PyCFunction)setMirrorMode, METH_VARARGS },
	{ "getCal", (PyCFunction)getCal, METH_VARARGS },
	{ "getDepthAccuracyDefault", (PyCFunction)getAccuracyDefault, METH_VARARGS },
	{ "getDepthFilterInfo", (PyCFunction)getFilterInfo, METH_VARARGS },
	{ "getDepthLaserPowerInfo", (PyCFunction)getLaserPowerInfo, METH_VARARGS },
	{ "getDepthRangeInfo", (PyCFunction)getRangeInfo, METH_VARARGS },
	{ "getCoordinateSystem", (PyCFunction)getCoordinateSystem, METH_VARARGS },
	{ "setCoordinateSystem", (PyCFunction)setCoordinateSystem, METH_VARARGS },


	{0,0,0}

	/*
	{ "", (PyCFunction), METH_VARARGS },
	*/
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