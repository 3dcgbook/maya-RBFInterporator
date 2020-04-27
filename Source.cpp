#include "Node.hpp"
#include "RbfInterporator.hpp"
#include <Eigen/LU>
#include <cmath>
#include <stdio.h>
#include <maya/MGlobal.h>
#include <maya/MString.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MStatus.h>
#include <maya/MArrayDataHandle.h>
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::PartialPivLU;

MTypeId RbfInterporatorNode::id(0x00005);
MObject RbfInterporatorNode::aInRbfType;
MObject RbfInterporatorNode::aInNormalized;
MObject RbfInterporatorNode::aInLambda;
MObject RbfInterporatorNode::aInRadius;
MObject RbfInterporatorNode::aInPosition;
MObject RbfInterporatorNode::aInPositionX;
MObject RbfInterporatorNode::aInPositionY;
MObject RbfInterporatorNode::aInPositionZ;
MObject RbfInterporatorNode::aInFeature;
MObject RbfInterporatorNode::aInFeatureX;
MObject RbfInterporatorNode::aInFeatureY;
MObject RbfInterporatorNode::aInFeatureZ;
MObject RbfInterporatorNode::aInFeatureOut;
MObject RbfInterporatorNode::aOutWeight;
MObject RbfInterporatorNode::aOutPureWeight;
MObject RbfInterporatorNode::aOutCalculate;

RbfInterporatorNode::RbfInterporatorNode() {}
RbfInterporatorNode::~RbfInterporatorNode() {}

rbf::RbfInterpolator::RbfInterpolator(const std::function<double(const double)>& rbf_kernel)
	: m_rbf_kernel(rbf_kernel)
{

}

void rbf::RbfInterpolator::SetRbfKernel(const std::function<double(const double)>& rbf_kernel)
{
	this->m_rbf_kernel = rbf_kernel;
}

void rbf::RbfInterpolator::SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
	assert(y.rows() == X.cols());
	this->m_X = X;
	this->m_y = y;
}

void rbf::RbfInterpolator::CalcWeights(const bool use_regularization, const double lambda)
{
	const int dim = m_y.rows();

	MatrixXd Phi = MatrixXd::Zero(dim, dim);
	for (int i = 0; i < dim; ++i)
	{
		for (int j = i; j < dim; ++j)
		{
			const double value = CalcRbfValue(m_X.col(i), m_X.col(j));
			Phi(i, j) = value;
			Phi(j, i) = value;
		}
	}

	const MatrixXd A = use_regularization ? Phi.transpose() * Phi + lambda * MatrixXd::Identity(dim, dim) : Phi;
	const VectorXd b = use_regularization ? Phi.transpose() * m_y : m_y;

	m_w = PartialPivLU<MatrixXd>(A).solve(b);
}

double rbf::RbfInterpolator::CalcValue(const VectorXd& x) const
{
	const int dim = m_w.rows();
	double result = 0.0;
	for (int i = 0; i < dim; ++i)
	{
		result += m_w(i) * CalcRbfValue(x, m_X.col(i));
	}

	return result;
}

inline double rbf::RbfInterpolator::CalcSingleValue(const VectorXd& x, const int& i) const
{
	return m_w(i) * CalcRbfValue(x, m_X.col(i));
}

double rbf::RbfInterpolator::CalcRbfValue(const VectorXd& xi, const VectorXd& xj) const
{
	assert(xi.rows() == xj.rows());
	return m_rbf_kernel((xj - xi).norm());
}

inline double rbf::RbfInterpolator::GetWeights(const int& i) const
{
	return m_w(i);
}

MStatus RbfInterporatorNode::compute(const MPlug& plug, MDataBlock& data)
{	
	MStatus status;
	/*
	if (plug != aOutWeight) {
		return MS::kUnknownParameter;
	}*/
	char moji[256];

	bool isNormalized = data.inputValue(aInNormalized).asBool();
	short rbftype = data.inputValue(aInRbfType).asShort();
	double lambda = data.inputValue(aInLambda).asDouble();
	double radius = data.inputValue(aInRadius).asDouble();
	MArrayDataHandle inArryHandle = data.inputArrayValue(aInFeature, &status);
	unsigned x_number_of_samples = inArryHandle.elementCount();

	MArrayDataHandle inYHandle = data.inputArrayValue(aInFeatureOut, &status);
	unsigned y_number_of_samples = inYHandle.elementCount();

	if (x_number_of_samples != y_number_of_samples)
	{
		MGlobal::displayInfo("x element count not equal y.");
		return MS::kUnknownParameter;
	}

	MatrixXd X(3, x_number_of_samples);
	Eigen::VectorXd y(x_number_of_samples);
	

	for (unsigned int i = 0; i < x_number_of_samples; i++)
	{
		inArryHandle.jumpToElement(i);
		double3& position = inArryHandle.inputValue().asDouble3();
		Eigen::Vector3d ePosition = Eigen::Vector3d(position[0], position[1], position[2]);
		if (isNormalized)
			ePosition.normalize();
		Eigen::Index top_col = i * X.rows();
		X(top_col) = ePosition(0);
		X(top_col+1) = ePosition(1);
		X(top_col+2) = ePosition(2);

		//y[i] 
		inYHandle.jumpToElement(i);
		y(i) = inYHandle.inputValue().asDouble();
	}

	const auto kernel = rbf::GaussianRbfKernel(radius);
	constexpr bool use_regularization = true;

	rbf::RbfInterpolator rbf_interpolator(kernel);
	switch (rbftype)
	{
	case 0:
		MGlobal::displayInfo("LinearRbfKernel");
		rbf_interpolator.SetRbfKernel(rbf::LinearRbfKernel());
		break;
	case 1:
		rbf_interpolator.SetRbfKernel(rbf::GaussianRbfKernel(radius));
		break;
	case 2:
		rbf_interpolator.SetRbfKernel(rbf::ThinPlateSplineRbfKernel());
		break;
	case 3:
		rbf_interpolator.SetRbfKernel(rbf::InverseQuadraticRbfKernel());
		break;
	default:
		break;
	}
	
	rbf_interpolator.SetData(X, y);
	rbf_interpolator.CalcWeights(use_regularization, lambda);

	double3& inpos = data.inputValue(aInPosition).asDouble3();
	Eigen::Vector3d input_x = Eigen::Vector3d(inpos[0], inpos[1], inpos[2]);
	if (isNormalized)
		input_x.normalize();
	double yy = rbf_interpolator.CalcValue(input_x);
	
	MArrayDataHandle hOutArr(data.outputArrayValue(aOutWeight));
	MArrayDataHandle pwOutArry(data.outputArrayValue(aOutPureWeight));
	unsigned int outWeight_count = hOutArr.elementCount();
	unsigned int outPuretWeight_count = pwOutArry.elementCount();
	if (outWeight_count != outPuretWeight_count)
	{
		MGlobal::displayInfo("weight element count not equal pureweight.");
		return MS::kUnknownParameter;
	}
	for (unsigned int i = 0; i < outWeight_count; i++)
	{
		hOutArr.jumpToElement(i);
		pwOutArry.jumpToElement(i);
		
		
		hOutArr.outputValue().setDouble(rbf_interpolator.CalcSingleValue(input_x, i));
		pwOutArry.outputValue().setDouble(rbf_interpolator.GetWeights(i));
		sprintf_s(moji, "%f", rbf_interpolator.GetWeights(i));
		//MGlobal::displayInfo(moji);
		hOutArr.setClean();
		pwOutArry.setClean();
	}
	//MGlobal::displayInfo("-----");

	data.outputValue(aOutCalculate).setDouble(yy);

	//MArrayDataHandle hOutArr(data.outputArrayValue(aOutWeight));
	return MS::kSuccess;
}

void* RbfInterporatorNode::creator()
{
	return new RbfInterporatorNode();
}

MStatus RbfInterporatorNode::initialize()
{
	MStatus stat;
	MFnNumericAttribute fnNum;
	MFnUnitAttribute fnUnit;
	MFnEnumAttribute fnEnum;

	aInNormalized = fnNum.create("normalized", "norm", MFnNumericData::kBoolean, false);


	aInRbfType = fnEnum.create("rbftype", "rt", 1, &stat);
	fnEnum.addField("Linear", 0);
	fnEnum.addField("Gaussian", 1);
	fnEnum.addField("ThinPlateSpline", 2);
	fnEnum.addField("InverseQuadratic", 3);
	
	aInLambda = fnNum.create("lambda", "l", MFnNumericData::kDouble, 0.001);
	aInRadius = fnNum.create("radius", "r", MFnNumericData::kDouble, 1.0);


	aInPositionX = fnUnit.create("inputX", "ix", MFnUnitAttribute::kDistance, 0.0);
	aInPositionY = fnUnit.create("inputY", "iy", MFnUnitAttribute::kDistance, 0.0);
	aInPositionZ = fnUnit.create("inputZ", "iz", MFnUnitAttribute::kDistance, 0.0);
	aInPosition = fnNum.create("input", "i", aInPositionX, aInPositionY, aInPositionZ);

	aInFeatureX = fnUnit.create("featureX", "fx", MFnUnitAttribute::kDistance, 0.0);
	aInFeatureY = fnUnit.create("featureY", "fy", MFnUnitAttribute::kDistance, 0.0);
	aInFeatureZ = fnUnit.create("featureZ", "fz", MFnUnitAttribute::kDistance, 0.0);
	aInFeature = fnNum.create("feature", "f", aInFeatureX, aInFeatureY, aInFeatureZ);
	fnNum.setArray(true);

	aInFeatureOut = fnNum.create("featureOut", "fo", MFnNumericData::kDouble, 0.0);
	fnNum.setArray(true);

	aOutWeight = fnNum.create("weight", "w", MFnNumericData::kDouble, 0.0);
	fnNum.setStorable(false);
	fnNum.setWritable(false);
	fnNum.setArray(true);

	aOutPureWeight = fnNum.create("pureweight", "pw", MFnNumericData::kDouble, 0.0);
	fnNum.setStorable(false);
	fnNum.setWritable(false);
	fnNum.setArray(true);

	aOutCalculate = fnNum.create("calculate", "calc", MFnNumericData::kDouble, 0.0);
	fnNum.setStorable(false);
	fnNum.setWritable(false);

	addAttribute(aInNormalized);
	addAttribute(aInRbfType);
	addAttribute(aInLambda);
	addAttribute(aInRadius);
	addAttribute(aInPosition);
	addAttribute(aInFeatureOut);
	addAttribute(aInFeature);
	addAttribute(aOutWeight);
	addAttribute(aOutPureWeight);
	addAttribute(aOutCalculate);

	attributeAffects(aInNormalized, aOutWeight);
	attributeAffects(aInRbfType, aOutWeight);
	attributeAffects(aInLambda, aOutWeight);
	attributeAffects(aInRadius, aOutWeight);
	attributeAffects(aInPosition, aOutWeight);
	attributeAffects(aInFeature, aOutWeight);
	attributeAffects(aInFeatureOut, aOutWeight);

	attributeAffects(aInNormalized, aOutCalculate);
	attributeAffects(aInRbfType, aOutCalculate);
	attributeAffects(aInLambda, aOutCalculate);
	attributeAffects(aInRadius, aOutCalculate);
	attributeAffects(aInPosition, aOutCalculate);
	attributeAffects(aInFeature, aOutCalculate);
	attributeAffects(aInFeatureOut, aOutCalculate);

	attributeAffects(aInNormalized, aOutPureWeight);
	attributeAffects(aInRbfType, aOutPureWeight);
	attributeAffects(aInLambda, aOutPureWeight);
	attributeAffects(aInRadius, aOutPureWeight);
	attributeAffects(aInPosition, aOutPureWeight);
	attributeAffects(aInFeature, aOutPureWeight);
	attributeAffects(aInFeatureOut, aOutPureWeight);

	return MS::kSuccess;

}