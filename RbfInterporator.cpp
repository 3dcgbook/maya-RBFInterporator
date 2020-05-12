#include "RbfInterporator.hpp"
#include "RbfInterporatorUtil.hpp"
#include <Eigen/LU>
#include <cmath>
#include <stdio.h>
#include <maya/MGlobal.h>
#include <maya/MString.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MStatus.h>
#include <maya/MArrayDataHandle.h>
#include <iostream>
#include <maya/MEvaluationNode.h>
#include <maya/MEvaluationManager.h>
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
MObject RbfInterporatorNode::aOutCalculate;
MObject RbfInterporatorNode::aOutWeights;
MObject RbfInterporatorNode::aOutBaseWeight;
MObject RbfInterporatorNode::aOutRBFWeight;
MObject RbfInterporatorNode::aInFeatures;
MObject RbfInterporatorNode::aInFeatureBase;
MObject RbfInterporatorNode::aInFeatureBaseX;
MObject RbfInterporatorNode::aInFeatureBaseY;
MObject RbfInterporatorNode::aInFeatureBaseZ;
MObject RbfInterporatorNode::aInFeatureOut;

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

RbfInterporatorNode::RbfInterporatorNode()
	: valueDirty(true)
{
}

RbfInterporatorNode::~RbfInterporatorNode()
{
}

MStatus RbfInterporatorNode::setDependentsDirty(const MPlug& plug, MPlugArray& plugArray)
{
	const MPlug dirtyPlug = plug.isChild() ? plug.parent() : plug;
	if (dirtyPlug == aInNormalized || dirtyPlug == aInRbfType || dirtyPlug == aInLambda || dirtyPlug == aInRadius) 
	{
		//MGlobal::displayInfo("setDependentsDirty: " + dirtyPlug.name());
		valueDirty = true;
	}
	return MPxNode::setDependentsDirty(plug, plugArray);
}

MStatus RbfInterporatorNode::preEvaluation(const MDGContext& context, const MEvaluationNode& evalNode)
{
	if (!valueDirty)
	{
		for (MEvaluationNodeIterator it = evalNode.iterator(); !it.isDone(); it.next()) {
			MPlug dirtyPlug = it.plug();
			if (dirtyPlug == aInPosition || dirtyPlug == aInFeatures) {
				//MGlobal::displayInfo("PreEval: " + dirtyPlug.name());
				valueDirty = true;
			}
		}
	}
	return MS::kSuccess;
}

MStatus RbfInterporatorNode::initialize()
{
	MStatus stat;
	MFnNumericAttribute fnNum;
	MFnUnitAttribute fnUnit;
	MFnEnumAttribute fnEnum;
	MFnCompoundAttribute fnComp;

	aInNormalized = fnNum.create("Normalized", "norm", MFnNumericData::kBoolean, false);
	aInRbfType = fnEnum.create("Rbftype", "rt", 1, &stat);
	fnEnum.addField("Linear", 0);
	fnEnum.addField("Gaussian", 1);
	fnEnum.addField("ThinPlateSpline", 2);
	fnEnum.addField("InverseQuadratic", 3);

	aInLambda = fnNum.create("Lambda", "l", MFnNumericData::kDouble, 0.001);
	aInRadius = fnNum.create("Radius", "r", MFnNumericData::kDouble, 1.0);

	aInPositionX = fnUnit.create("InputX", "ix", MFnUnitAttribute::kDistance, 0.0);
	aInPositionY = fnUnit.create("InputY", "iy", MFnUnitAttribute::kDistance, 0.0);
	aInPositionZ = fnUnit.create("InputZ", "iz", MFnUnitAttribute::kDistance, 0.0);
	aInPosition = fnNum.create("Input", "i", aInPositionX, aInPositionY, aInPositionZ);

	aOutCalculate = fnNum.create("Calculate", "calc", MFnNumericData::kDouble, 0.0);
	fnNum.setStorable(false);
	fnNum.setWritable(false);

	// コンパウンド型アウトプットのテスト用
	aOutBaseWeight = fnNum.create("BaseWeight", "bw", MFnNumericData::kDouble, 0.0);
	aOutRBFWeight = fnNum.create("RbfWeight", "rw", MFnNumericData::kDouble, 0.0);

	aOutWeights = fnComp.create("OutWeights", "ow");
	fnComp.addChild(aOutBaseWeight);
	fnComp.addChild(aOutRBFWeight);
	fnComp.setArray(true);
	fnComp.setWritable(false);
	fnComp.setStorable(false);

	// コンパウンド型インプットのテスト用
	aInFeatureBaseX = fnUnit.create("FeatureX", "fx", MFnUnitAttribute::kDistance, 0.0, &stat);
	aInFeatureBaseY = fnUnit.create("FeatureY", "fy", MFnUnitAttribute::kDistance, 0.0, &stat);
	aInFeatureBaseZ = fnUnit.create("FeatureZ", "fz", MFnUnitAttribute::kDistance, 0.0, &stat);
	aInFeatureBase = fnNum.create("Feature", "f", aInFeatureBaseX, aInFeatureBaseY, aInFeatureBaseZ, &stat);

	aInFeatureOut = fnNum.create("FeatureOut", "fo", MFnNumericData::kDouble, 0.0, &stat);

	aInFeatures = fnComp.create("Features", "Features", &stat);
	fnComp.addChild(aInFeatureBase);
	fnComp.addChild(aInFeatureOut);
	fnComp.setArray(true);

	addAttribute(aInNormalized);
	addAttribute(aInRbfType);
	addAttribute(aInLambda);
	addAttribute(aInRadius);
	addAttribute(aInPosition);
	addAttribute(aOutCalculate);
	addAttribute(aOutWeights);
	addAttribute(aInFeatures);

	attributeAffects(aInNormalized, aOutWeights);
	attributeAffects(aInPosition, aOutWeights);
	attributeAffects(aInRbfType, aOutWeights);
	attributeAffects(aInLambda, aOutWeights);
	attributeAffects(aInRadius, aOutWeights);
	attributeAffects(aInFeatures, aOutWeights);
	
	attributeAffects(aInNormalized, aOutCalculate);
	attributeAffects(aInRbfType, aOutCalculate);
	attributeAffects(aInLambda, aOutCalculate);
	attributeAffects(aInRadius, aOutCalculate);
	attributeAffects(aInPosition, aOutCalculate);
	attributeAffects(aInFeatures, aOutCalculate);

	return MS::kSuccess;

}

MStatus RbfInterporatorNode::compute(const MPlug& plug, MDataBlock& data)
{	
	MStatus status;
	double3& inpos = data.inputValue(aInPosition).asDouble3();
	MArrayDataHandle outArryHandle(data.outputArrayValue(aOutWeights));
	unsigned out_count = outArryHandle.elementCount();
	if (out_count == 0)
		return MS::kUnknownParameter;

	MArrayDataHandle featureArryHandle = data.inputArrayValue(aInFeatures, &status);
	unsigned feature_count = featureArryHandle.elementCount();

	if (feature_count != out_count){
		MGlobal::displayWarning("feature count is not equal to out count.");
		return MS::kUnknownParameter;
	}

	bool isNormalized = data.inputValue(aInNormalized).asBool();
	short rbftype = data.inputValue(aInRbfType).asShort();
	double lambda = data.inputValue(aInLambda).asDouble();
	double radius = data.inputValue(aInRadius).asDouble();


	if (valueDirty) {
		//MGlobal::displayInfo("Compute Dirty");
		MatrixXd X(3, feature_count);
		VectorXd y(feature_count);
	
		for (unsigned i = 0; i < feature_count; i++) {
			featureArryHandle.jumpToElement(i);

			// XYZのプロット値の取得
			double3& position = featureArryHandle.inputValue(&status).child(aInFeatureBase).asDouble3();
			Eigen::Vector3d eg_pos = Eigen::Vector3d(position[0], position[1], position[2]);
			if (isNormalized)
				eg_pos.normalize();
			Eigen::Index column = i * X.rows();
			X(column) = eg_pos(0);
			X(column + 1) = eg_pos(1);
			X(column + 2) = eg_pos(2);

			// プロット結果の取得
			y(i) = featureArryHandle.inputValue(&status).child(aInFeatureOut).asDouble();
			Eigen::Vector3d ePosition = Eigen::Vector3d(position[0], position[1], position[2]);
		}
		// RBF Interporationの実装
		const auto kernel = rbf::GaussianRbfKernel(radius);
		constexpr bool use_regularization = true;
		pRbf.SetRbfKernel(kernel);

		switch (rbftype)
		{
		case 0:
			pRbf.SetRbfKernel(rbf::LinearRbfKernel());
			break;
		case 1:
			pRbf.SetRbfKernel(rbf::GaussianRbfKernel(radius));
			break;
		case 2:
			pRbf.SetRbfKernel(rbf::ThinPlateSplineRbfKernel());
			break;
		case 3:
			pRbf.SetRbfKernel(rbf::InverseQuadraticRbfKernel());
			break;
		default:
			break;
		}
		pRbf.SetData(X, y);
		pRbf.CalcWeights(use_regularization, lambda);
		valueDirty = false;
	}
	
	Eigen::Vector3d input_x = Eigen::Vector3d(inpos[0], inpos[1], inpos[2]);
	if (isNormalized)
		input_x.normalize();
	double yy = pRbf.CalcValue(input_x);
	//char moji[256];
	//sprintf_s(moji, "CalculateOut : %f", inpos[0]);
	//MGlobal::displayInfo(moji);
	for (unsigned i = 0; i < out_count; i++) {
		outArryHandle.jumpToElement(i);
		outArryHandle.outputValue(&status).child(aOutRBFWeight).setDouble(pRbf.CalcSingleValue(input_x, i));
		outArryHandle.outputValue(&status).child(aOutBaseWeight).setDouble(pRbf.GetWeights(i));
		outArryHandle.setClean();
	}
	data.outputValue(aOutCalculate).setDouble(yy);
	data.setClean(plug);
	return MS::kSuccess;
}

void* RbfInterporatorNode::creator()
{
	return new RbfInterporatorNode();
}
