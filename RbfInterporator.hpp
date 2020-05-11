#pragma once
#include <maya/MPxNode.h>

class RbfInterporatorNode : public MPxNode
{
public:
	RbfInterporatorNode();
	virtual ~RbfInterporatorNode();
	virtual MStatus compute(const MPlug& plug, MDataBlock& data);
	static void* creator();
	static MStatus initialize();
public:
	static MTypeId id;
	static MObject aInNormalized;
	static MObject aInRbfType;
	static MObject aInLambda;
	static MObject aInRadius;
	static MObject aInPosition;
	static MObject aInPositionX;
	static MObject aInPositionY;
	static MObject aInPositionZ;

	static MObject aOutWeights;
	static MObject aOutBaseWeight;
	static MObject aOutRBFWeight;

	static MObject aInFeatures;
	static MObject aInFeatureBase;
	static MObject aInFeatureBaseX;
	static MObject aInFeatureBaseY;
	static MObject aInFeatureBaseZ;
	static MObject aInFeatureOut;


	static MObject aOutCalculate;
};