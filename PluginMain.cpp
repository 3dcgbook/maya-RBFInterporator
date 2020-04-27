#include "Node.hpp"
#include <maya/MFnPlugin.h>

MStatus initializePlugin(MObject obj)
{
	MStatus status;
	MFnPlugin plugin(obj, "Shuhei Arai", "2020", "Any");

	status = plugin.registerNode("RBFInterporator", RbfInterporatorNode::id, RbfInterporatorNode::creator, RbfInterporatorNode::initialize);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	return status;
}

MStatus uninitializePlugin(MObject obj)
{
	MStatus status;
	MFnPlugin plugin(obj);
	status = plugin.deregisterNode(RbfInterporatorNode::id);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	return status;
}