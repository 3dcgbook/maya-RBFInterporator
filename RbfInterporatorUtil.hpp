#pragma once
#include <Eigen/Core>

namespace rbf
{
	class AbstractRbfKernel
	{
	public:
		AbstractRbfKernel() {}
		virtual ~AbstractRbfKernel() {};

		virtual double operator()(const double r) const = 0;
	};

	class GaussianRbfKernel final : public AbstractRbfKernel
	{
	public:
		GaussianRbfKernel(const double theta = 1.0) : m_theta(theta) {}

		double operator()(const double r) const override
		{
			assert(r >= 0.0);
			return std::exp(-m_theta * r * r);
		}
	private:
		const double m_theta;
	};

	class ThinPlateSplineRbfKernel final : public AbstractRbfKernel
	{
	public:
		ThinPlateSplineRbfKernel() {}

		double operator()(const double r) const override
		{
			assert(r >= 0.0);
			const double value = r * r * std::log(r);
			return std::isnan(value) ? 0.0 : value;
		}

	};

	class LinearRbfKernel final : AbstractRbfKernel
	{
	public:
		LinearRbfKernel() {}
		double operator()(const double r) const override { return std::abs(r); }
	};

	class InverseQuadraticRbfKernel final : public AbstractRbfKernel
	{
	public:
		InverseQuadraticRbfKernel(const double theta = 1.0) : m_theta(theta) {}
		double operator()(const double r) const override { return 1.0 / std::sqrt(r * r + m_theta * m_theta); }

	private:
		const double m_theta;
	};

	class RbfInterpolator
	{
	public:
		RbfInterpolator(const std::function<double(const double)>& rbf_kernel = ThinPlateSplineRbfKernel());

		void SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
		void CalcWeights(const bool use_regularization = false, const double lambda = 0.001);
		void SetRbfKernel(const std::function<double(const double)>& rbf_kernel);
		double CalcValue(const Eigen::VectorXd& x) const;
		double CalcSingleValue(const Eigen::VectorXd& x, const int& i) const;
		double GetWeights(const int& i) const;
		
		
		
	private:
		std::function<double(double) > m_rbf_kernel;
		Eigen::MatrixXd m_X;
		Eigen::VectorXd m_y;
		Eigen::VectorXd m_w;


		double CalcRbfValue(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const;
	};
}