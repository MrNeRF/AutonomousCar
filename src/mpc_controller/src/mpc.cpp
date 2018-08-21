#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <qpOASES.hpp>
#include "ros/ros.h"
#include <custom_msg/mpc_control.h>
#include <custom_msg/car_position.h>
#include <custom_msg/sensor_measures.h>
#include <custom_msg/optimize.h>
#include <custom_msg/track.h>
#include "std_msgs/Float64MultiArray.h"
#include "visualization_msgs/Marker.h"
#include  "nav_msgs/Odometry.h"

// define PI
constexpr double pi() {return std::atan(1) * 4;}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

class MPC_Controller {
	public: 
		MPC_Controller(ros::NodeHandle &nh_, int horizon_, Eigen::MatrixXd Q_,\
				Eigen::MatrixXd R_, Eigen::MatrixXd track_, double vel, Eigen::Vector2d umin, \
				Eigen::Vector2d umax, int nx_, int nu_, double c_) : ekf_state(3), ekf_bar(3){

			nh = nh_;
			horizon = horizon_;
			Q = Q_;
			R = R_;
			u_min = umin;
			u_max = umax;
			nx = nx_;
			u_opt << 0,0;
			nu = nu_;
			target_velocity = vel;
			track = track_;
			c = c_;
			dt = (double)1.0/(double)30.0;
			number_refpoints = track.cols();
			x_input = Eigen::MatrixXd::Zero(nx, horizon);
			u_input = Eigen::MatrixXd::Zero(nu, horizon);
			car_pos << 0, 0, 0;
			currentIndex = 0;
			
			// Odomotrie Calibtration
			delta_heading = Eigen::MatrixXd::Zero(1,1);
			delta_wheels  = Eigen::MatrixXd::Zero(1,2);
			delta_track   = Eigen::MatrixXd::Zero(1,1);
			phi_track     = Eigen::MatrixXd::Zero(1,1);
			deltaAngleLeftWheel = 0.0;
			deltaAngleRightWheel = 0.0;


			kalman_index = 0;
			factor = 0.0;
			firstrun = true;
			time_begin = std::chrono::high_resolution_clock::now();
			// Kalman filter init

			Hk = Eigen::MatrixXd::Zero(6,3);
			Hk(0,0) = 1.0, Hk(1,1) = 1.0, Hk(2,2) = 1.0, Hk(3,0) = 1.0, Hk(4,1) = 1.0, Hk(5,2) = 1.0;
			Rk = Eigen::MatrixXd::Identity(6,6);
			Rk(0,0) = .5, Rk(1,1) = .5, Rk(2,2) = 0.1, Rk(3,3) = .3, Rk(4,4) = .3, Rk(5,5) = .5;
			Pk = Eigen::MatrixXd::Identity(3,3);
			Pk(0,0) = .0, Pk(1,1) = .0; Pk(2,2) = .0;
			Qk = Eigen::MatrixXd::Identity(3, 3);
			Qk(0,0) = 1., Qk(1,1) = 1., Qk(2,2) = 0.1; 
			Qk(2,0) = Qk(0,2) = 0.0;
			Qk(1,0) = Qk(0,1) = 0.0;
			Qk(1,2) = Qk(2,1) = 0.0;

			ekf_state << 1.0, 1.0, 1.0;
			ekf_bar << 0.0, 0.0, 0.0;
			model <<  0.0, 0.0, 0.0;
			imu_x_measured = 0.0;
			imu_y_measured = 0.0;
			imu_yaw_measured = 0.0;
			imu_delta_yaw_measured = 0.0;
			lin_vel_measured = 0.0;
			odom_x_measured = 0.0;
			odom_y_measured = 0.0;
			odom_yaw_measured = 0.0;
			odom_delta_yaw_measured = 0.0;
				
			positionSubscriber = nh.subscribe("car_position", 10, &MPC_Controller::car_position_cb, this);
			measurementSubscriber = nh.subscribe("sensor_measures", 500, &MPC_Controller::measurement_cb, this);
			pubMPC = nh.advertise<custom_msg::mpc_control>("mpc_control", 1);
			pubOptimize = nh.advertise<custom_msg::optimize>("optimize", 1);
			pub_mpcmarker = nh.advertise<visualization_msgs::Marker>("mpc_marker", 10);
		}

	private:
		bool firstrun;
		Eigen::Vector2d u_opt;
		int number_refpoints;
		int horizon;
		int nx, nu;
		double target_velocity;
		double c;
		double dt;
		std::chrono::high_resolution_clock::time_point time_begin;
		int currentIndex, kalman_index;
		double factor;
		ros::NodeHandle nh;
		ros::Subscriber positionSubscriber, measurementSubscriber;
		ros::Publisher pubMPC, pub_mpcmarker, pubOptimize;
		Eigen::Vector3d car_pos;
		Eigen::Vector3d init_state;
		Eigen::Vector2d u_min;
		Eigen::Vector2d u_max;
		Eigen::MatrixXd track;
		Eigen::MatrixXd Q;
		Eigen::MatrixXd R;
		Eigen::MatrixXd x_input;
		Eigen::MatrixXd u_input;
		Eigen::MatrixXd delta_heading;
		Eigen::MatrixXd delta_wheels;
		Eigen::MatrixXd delta_track;
		Eigen::MatrixXd phi_track;
		double deltaAngleLeftWheel; 
		double deltaAngleRightWheel;
		double calibHeading;
		//create blk diagonal matrix
		void car_position_cb(const custom_msg::car_position::ConstPtr& p);
		void measurement_cb(const custom_msg::sensor_measures::ConstPtr& measure);
		void calc_mpc(Eigen::Vector3d state, bool sentMarker);
		double angleDifference(double diff);
		int getQuadrant(double quadrant);
		//void car_position_cb(const nav_msgs::Odometry::ConstPtr& p);
		Eigen::Vector3d wmr(Eigen::Vector3d x, Eigen::Vector2d u, double stepsize_h);
		Eigen::VectorXd mpc_controller(Eigen::MatrixXd x_ref, Eigen::MatrixXd u_ref, Eigen::Vector3d state, bool sentMarker);
		Eigen::MatrixXd A_k(Eigen::MatrixXd u_ref, Eigen::MatrixXd x_ref, int i);
		Eigen::MatrixXd B_k(Eigen::MatrixXd x_ref, int i);
		Eigen::MatrixXd alpha(Eigen::MatrixXd &u_ref, Eigen::MatrixXd &x_ref, 
				int start, int stop, std::vector<long> &dim);
		Eigen::MatrixXd A_bar(Eigen::MatrixXd &u_ref, Eigen::MatrixXd &x_ref, 
				std::vector<long> &dim);
		Eigen::MatrixXd B_bar(Eigen::MatrixXd &u_ref, Eigen::MatrixXd &x_ref, 
				std::vector<long> dimA, std::vector<long> dimB);
		Eigen::MatrixXd blkdiag(const Eigen::MatrixXd& a, int count); 
		Eigen::MatrixXd calc_beta(Eigen::MatrixXd &xref, Eigen::MatrixXd &uref, int i, 
				int j, std::vector<long> &dimA);
		int getRefXandU(int index, Eigen::VectorXd state);
		int quadTransTo(double first, double second);
		void sendMPCMarker(Eigen::MatrixXd nextGoal, Eigen::MatrixXd mpcdata);

		// Kalman Filter Methods
		Eigen::MatrixXd Pk, Qk, Hk, Rk;
		Eigen::VectorXd ekf_state;
		Eigen::VectorXd ekf_bar;
		double imu_x_measured;
		double imu_y_measured; 
		double imu_yaw_measured;
		double imu_delta_yaw_measured;
		double lin_vel_measured;
		double odom_delta_yaw_measured;
		double odom_x_measured;
		double odom_y_measured;
		double odom_yaw_measured;
		Eigen::Vector3d model;
		void extended_KF(double delta_t);
};

Eigen::Vector3d MPC_Controller::wmr(Eigen::Vector3d x, Eigen::Vector2d u, double stepsize_h){
	Eigen::Vector3d newstate;
	newstate << 0.0,0.0,0.0;
	newstate(0) = x(0) + stepsize_h * u(0) * cos(x(2));
	newstate(1) = x(1) + stepsize_h * u(0) * sin(x(2));
	newstate(2) = x(2) + stepsize_h * u(1);
	return newstate;
}


void MPC_Controller::measurement_cb(const custom_msg::sensor_measures::ConstPtr& measure) {
	if(firstrun) 
		return;

	lin_vel_measured			= measure->lin_vel;
	imu_delta_yaw_measured		= measure->imu_yaw_delta;
	odom_delta_yaw_measured		= measure->odom_yaw;
	deltaAngleLeftWheel			+= measure->deltaAngleLeftWheel;
	deltaAngleRightWheel		+= measure->deltaAngleRightWheel;
	double delta_t = measure->dt;

	deltaAngleLeftWheel = angleDifference(deltaAngleLeftWheel);
	deltaAngleRightWheel = angleDifference(deltaAngleRightWheel);

	extended_KF(delta_t);

}

void MPC_Controller::sendMPCMarker(Eigen::MatrixXd nextGoal, Eigen::MatrixXd mpcdata) {
	visualization_msgs::Marker points;
	points.header.frame_id = "/odom";
	points.header.stamp = ros::Time::now();
	points.id = 0;

	points.type = visualization_msgs::Marker::POINTS;
	// POINTS markers use x and y scale for width/height respectively
	points.scale.x = 0.03;
	points.scale.y = 0.03;
	// Points are green
	points.color.r = 1.0f;
	points.color.a = 1.0;

	// Create the vertices for the points and lines
	geometry_msgs::Point g;
	g.x = nextGoal(0,0);
	g.y = nextGoal(1,0);
	points.points.push_back(g);

	for (int i = 0; i < mpcdata.cols(); ++i)
	{
		geometry_msgs::Point p;
		p.x = mpcdata(0,i);
		p.y = mpcdata(1,i);

		points.points.push_back(p);

	}
	points.action = visualization_msgs::Marker::DELETEALL;
	pub_mpcmarker.publish(points);
	points.action = visualization_msgs::Marker::ADD;
	pub_mpcmarker.publish(points);
}


int MPC_Controller::getRefXandU(int index, Eigen::VectorXd state) {
	Eigen::MatrixXd uref = Eigen::MatrixXd::Zero(nu, horizon);
	Eigen::MatrixXd xref = Eigen::MatrixXd::Zero(nx, horizon + 1);
	double lad = 0.25;  // lookahead distance
	double distance = 0.1;
	volatile double norm;
	int discretization = 80;
	int tmp_index = index;


	// prepare lookahead half circle
	Eigen::ArrayXXd lahead_circle(2, discretization);
	lahead_circle.row(0) = Eigen::ArrayXd::LinSpaced(discretization, state(2) - 0.5 * pi(), state(2) + 0.5 * pi());
	lahead_circle.row(1) = Eigen::ArrayXd::LinSpaced(discretization, state(2) - 0.5 * pi(), state(2) + 0.5 *pi());
	lahead_circle.row(0) = lahead_circle.row(0).cos() * lad;
	lahead_circle.row(1) = lahead_circle.row(1).sin() * lad;
	Eigen::MatrixXd car_circle = lahead_circle.matrix();
	car_circle = car_circle.colwise() + state.block(0,0,2,1);

	for(int i =0; i < discretization; i++)	 {
		for(int j = index; j < index + 80; j++) {
			int l = j % number_refpoints;

			double x = car_circle(0,i) - track(0,l);
			double y = car_circle(1,i) - track(1,l);
			norm = std::sqrt(x*x+y*y);

			if(norm < distance) {
				distance = norm;
				tmp_index= l;
			}
		}
	}


	index = tmp_index;

	Eigen::VectorXd d = track.block(0,index,2,1) - state.block(0,0,2,1);
	double xv  = -d(0) * std::sin(state(2)) + d(1) * std::cos(state(2));
	norm = std::sqrt(d.dot(d));
	double curvature = 2 * xv / (norm * norm);

	xref.block(0,0,3,1) = state; 
	for(int n=0; n < horizon; n++){
		uref(0,n) = target_velocity;
		uref(1,n) = 2 * target_velocity * curvature;
		xref.block(0,n+1, 3,1) = wmr(xref.block(0,n,3,1), uref.block(0,n,2,1), dt);
		int currentQuadrant = getQuadrant(xref(2,n));
		if (currentQuadrant == 2 && quadTransTo(xref(2,n), xref(2,n+1)) == 3) {
			xref(2,n+1) += 2 * pi();
		}
		else if (currentQuadrant == 3 && quadTransTo(xref(2,n), xref(2,n+1)) == 2) {
			xref(2,0) -= 2 * pi();
		}
		state = xref.block(0,n+1,3,1);
	}
	x_input = xref;
	u_input = uref;
	return index % number_refpoints;
}


int MPC_Controller::getQuadrant(double angle) {
	if (angle >= 0.0 && angle <= pi()/2)
		return 1;
	else if(angle > pi()/2 && angle <= pi()) 
		return 2;
	else if(angle >= -pi() && angle < -pi()/2)
		return 3;
	else 
		return 4;
}
int MPC_Controller::quadTransTo(double first, double second) {
	if (getQuadrant(first) == 2 && getQuadrant(second) == 3) 
		return 3; // Transition from second to third quadrant
	else if (getQuadrant(first) == 3 && getQuadrant(second) == 2)
		return 2; // Transition from third to second quadrant
	return false;
}

double MPC_Controller::angleDifference(double diff) {
	if (diff > pi())
		diff -= 2 * pi();
	else if (diff < -pi())
		diff += 2 * pi();
	return diff;
}

void MPC_Controller::calc_mpc(Eigen::Vector3d state, bool sentMarker) {
	u_opt = mpc_controller(x_input, u_input, state, sentMarker);
}

void MPC_Controller::car_position_cb(const custom_msg::car_position::ConstPtr& p) {

	auto now = std::chrono::high_resolution_clock::now();
	dt  = std::chrono::duration<double, std::milli>(now - time_begin).count();
	time_begin = std::chrono::high_resolution_clock::now();
	dt = dt / 1000.0;

	Eigen::Vector3d state;
	state << p->xpos, p->ypos, p->yaw;
	car_pos = state;

	if(firstrun) {
		ekf_state(0) = imu_x_measured   = model(0) = odom_x_measured   = init_state(0) = state(0);
		ekf_state(1) = imu_y_measured   = model(1) = odom_y_measured   = init_state(1) = state(1);
		ekf_state(2) = imu_yaw_measured = model(2) = odom_yaw_measured = init_state(2) = state(2);
		firstrun = false;
	}

	currentIndex = getRefXandU(currentIndex, state);
	calc_mpc(state, true);

	model(0) = model(0) + u_opt(0) * cos(model(2) + dt * u_opt(1)) * dt;
	model(1) = model(1) + u_opt(0) * sin(model(2) + dt * u_opt(1)) * dt;
	model(2) = model(2) + u_opt(1) * dt;

	if (model(2) > pi())
		model(2) -= 2 * pi();
	else if (model(2) < -pi())
		model(2) += 2 * pi();

	custom_msg::mpc_control control;
	control.lin_vel	= u_opt(0);
	control.ang_vel	= u_opt(1);
	
	control.ekf_x		= ekf_state(0);
	control.ekf_y   	= ekf_state(1);
	control.ekf_yaw 	= ekf_state(2);
	control.imu_x		= imu_x_measured;
	control.imu_y 		= imu_y_measured;
	control.imu_yaw		= imu_yaw_measured;
	control.odom_x		= odom_x_measured;
	control.odom_y 		= odom_y_measured;
	control.odom_yaw	= odom_yaw_measured;
	control.state_x		= state(0);
	control.state_y 	= state(1);
	control.state_yaw	= state(2);
	control.ekf_bar_x 	= ekf_bar(0);
	control.ekf_bar_y 	= ekf_bar(1);
	control.ekf_bar_yaw = ekf_bar(2);
	control.model_x		= model(0);
	control.model_y		= model(1);
	control.model_yaw	= model(2);

	pubMPC.publish(control);
}

Eigen::MatrixXd MPC_Controller::blkdiag(const Eigen::MatrixXd& a, int count) {
	Eigen::MatrixXd bdm = Eigen::MatrixXd::Zero(a.rows() * count, a.cols() * count);
	for(int i = 0; i < count; ++i) {
		bdm.block(i* a.rows(), i* a.cols(),a.rows(), a.cols()) = a;
	}
	return bdm;
}

Eigen::VectorXd  MPC_Controller::mpc_controller(Eigen::MatrixXd x_ref, Eigen::MatrixXd u_ref, Eigen::Vector3d state, bool sentMarker){

	USING_NAMESPACE_QPOASES;
	Eigen::MatrixXd Q_bar = blkdiag(Q, horizon);
	Eigen::MatrixXd R_bar = blkdiag(R, horizon);
	// Reshape u_ref vector
	Eigen::Map<Eigen::VectorXd> uref_reshape(u_ref.data(), u_ref.size());
	Eigen::VectorXd u_min_ = u_min.replicate(horizon, 1) - uref_reshape;
	Eigen::VectorXd u_max_ = u_max.replicate(horizon, 1) - uref_reshape;

	// Generating Matrices to get the correct dimensions of both
	Eigen::MatrixXd A = A_k(u_ref, x_ref, 0);
	Eigen::MatrixXd B = B_k(x_ref, 0);
	std::vector<long> dimA {A.rows(), A.cols()}; 
	std::vector<long> dimB {B.rows(), B.cols()}; 

	Eigen::MatrixXd Ak_bar = A_bar(u_ref, x_ref, dimA);
	Eigen::MatrixXd Bk_bar = B_bar(u_ref, x_ref, dimA, dimB);

	Eigen::MatrixXd UBA(2,1);
	Eigen::MatrixXd LBA(2,1);
	Eigen::MatrixXd Atmp(2,2);
	Atmp << 1, -c, 1, c;
	Eigen::MatrixXd ALEQ = Atmp.replicate(1,horizon);
	UBA << target_velocity, target_velocity;
	LBA <<-target_velocity, -target_velocity;

	Eigen::MatrixXd x_tilda = state - x_ref.block(0,0,x_ref.rows(),1);

	// Cost function computation
	Eigen::MatrixXd HK = 2 * (Bk_bar.transpose() * Q_bar * Bk_bar + R_bar);
	Eigen::MatrixXd FK = 2 * Bk_bar.transpose() * Q_bar * Ak_bar * x_tilda;

	// convert everything to qpoases format
	real_t *hk = new real_t[HK.rows() * HK.cols()];
	real_t *fk = new real_t[FK.rows() * FK.cols()];
	//
	real_t *aleq = new real_t[ALEQ.rows() * ALEQ.cols()];
	real_t *lbA = new real_t[LBA.rows() * LBA.cols()];
	real_t *ubA = new real_t[UBA.rows() * UBA.cols()];
	//
	real_t *ub = new real_t[u_max_.rows() * u_max_.cols()];
	real_t *lb = new real_t[u_min_.rows() * u_min_.cols()];

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hk_qp(HK);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fk_qp(FK);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> aleq_qp(ALEQ);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lbA_qp(LBA);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ubA_qp(UBA);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> umin_qp(u_min_);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> umax_qp(u_max_);

	Eigen::Map<Eigen::RowVectorXd> hk_v(hk_qp.data(), hk_qp.size());
	Eigen::Map<Eigen::RowVectorXd> fk_v(fk_qp.data(), fk_qp.size());
	Eigen::Map<Eigen::RowVectorXd> aleq_v(aleq_qp.data(),aleq_qp.size());
	Eigen::Map<Eigen::RowVectorXd> lbA_v(lbA_qp.data(),lbA_qp.size());
	Eigen::Map<Eigen::RowVectorXd> ubA_v(ubA_qp.data(),ubA_qp.size());
	Eigen::Map<Eigen::RowVectorXd> umin_v(umin_qp.data(),umin_qp.size());
	Eigen::Map<Eigen::RowVectorXd> umax_v(umax_qp.data(), umax_qp.size());

	Eigen::Map<Eigen::RowVectorXd>(hk, hk_v.rows(), hk_v.cols()) = hk_v;
	Eigen::Map<Eigen::RowVectorXd>(fk, fk_v.rows(), fk_v.cols()) = fk_v;
	Eigen::Map<Eigen::RowVectorXd>(aleq, aleq_v.rows(), aleq_v.cols()) = aleq_v;
	Eigen::Map<Eigen::RowVectorXd>(ubA, ubA_v.rows(), ubA_v.cols()) = ubA_v;
	Eigen::Map<Eigen::RowVectorXd>(lb, umin_v.rows(), umin_v.cols()) = umin_v; 
	Eigen::Map<Eigen::RowVectorXd>(ub, umax_v.rows(), umax_v.cols()) = umax_v;


	int_t nWSR = 100;
	QProblem mpc_solver(HK.cols(), ALEQ.rows());
	// Options for the solver
	Options mpc_solver_opts;
	mpc_solver_opts.printLevel = PL_NONE;
	mpc_solver.setOptions(mpc_solver_opts);
	mpc_solver.init(hk, fk, aleq, lb, ub, lbA, ubA, nWSR);

	real_t *uopt = new real_t[HK.cols()];
	mpc_solver.getPrimalSolution(uopt);

	Eigen::VectorXd uk_bar = Eigen::Map<Eigen::VectorXd> (uopt, HK.rows(),1);
	Eigen::VectorXd xk_bar = Ak_bar * x_tilda + Bk_bar * uk_bar;
	Eigen::Map<Eigen::MatrixXd> xk_bar_reshape(xk_bar.data(), nx, horizon);
	Eigen::MatrixXd xk = xk_bar_reshape.block(0,0,nx,horizon) + x_ref.block(0,0,nx,horizon);

	if(sentMarker) {
		sendMPCMarker(track.block(0,currentIndex,2,1), xk);
	}

	delete [] uopt;
	delete [] hk ;
	delete [] fk ;

	return (uk_bar.block(0,0,2,1) + u_ref.block(0,0,2,1));
}

Eigen::MatrixXd MPC_Controller::A_k(Eigen::MatrixXd u_ref, Eigen::MatrixXd x_ref, int i) {
	Eigen::Matrix3d A_k;
	A_k << 1, 0, -u_ref(0,i) * std::sin(x_ref(2,i)) * dt,
		0, 1, u_ref(0,i) * std::cos(x_ref(2,i)) * dt, 
		0, 0, 1;
	return A_k;
}

Eigen::MatrixXd MPC_Controller::B_k(Eigen::MatrixXd x_ref, int k) {
	Eigen::MatrixXd B_k(3,2);
	B_k << std::cos(x_ref(2,k)) * dt, 0,
		std::sin(x_ref(2,k)) * dt, 0,
		0, dt;
	return B_k;
}

Eigen::MatrixXd MPC_Controller::calc_beta(Eigen::MatrixXd &xref, Eigen::MatrixXd &uref, int i,
		int j, std::vector<long> &dimA) {
	return alpha(uref, xref, i+1, j, dimA) * B_k(xref, i);
}

Eigen::MatrixXd MPC_Controller::alpha(Eigen::MatrixXd &u_ref, Eigen::MatrixXd &x_ref, 
		int start, int stop, std::vector<long> &dimA) {
	Eigen::MatrixXd m = Eigen::MatrixXd::Identity(dimA[0], dimA[0]);
	for(int k = start; k <= stop; ++k) {
		m = A_k(u_ref, x_ref, k) * m;
	}
	return m;
}

Eigen::MatrixXd MPC_Controller::A_bar(Eigen::MatrixXd &u_ref, Eigen::MatrixXd &x_ref, std::vector<long> &dimA) {
	Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dimA[0] * horizon, dimA[1]);
	for(int i = 0; i < horizon; ++i) {
		M.block(i * dimA[0], 0,dimA[0], dimA[1]) = alpha(u_ref, x_ref, i, 0, dimA);
	}

	return M;
}

Eigen::MatrixXd MPC_Controller::B_bar(Eigen::MatrixXd &u_ref, Eigen::MatrixXd &x_ref, 
		std::vector<long> dimA, std::vector<long> dimB) {
	Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dimB[0] * horizon, dimB[1] * horizon);
	for(int i=0; i < horizon; ++i) {
		for(int j = 0; j <= i; ++j) {
			if (i == j) {
				M.block(i * dimB[0], i * dimB[1],dimB[0], dimB[1]) = B_k(x_ref, j);
			} else {
				M.block(i * dimB[0], j * dimB[1],dimB[0], dimB[1]) = calc_beta(x_ref, u_ref, j, i, dimA);
			}
		}
	}
	return M;
}


void MPC_Controller::extended_KF(double delta_t) {

	/* Data for analysis with python */
	custom_msg::optimize optim;
	optim.odom_delta_yaw_measured = odom_delta_yaw_measured;
	optim.imu_delta_yaw_measured = imu_delta_yaw_measured;
	optim.linv_vel_measured = lin_vel_measured;
	optim.deltaAngleLeftWheel = deltaAngleLeftWheel;
	optim.deltaAngleRightWheel = deltaAngleRightWheel;
	optim.optLinVel = u_opt(0);
	optim.optAngVel = u_opt(1);
	optim.init_x   = init_state(0);
	optim.init_y   = init_state(1);
	optim.init_yaw = init_state(2);
	optim.current_x	  = car_pos(0); 
	optim.current_y   = car_pos(1);
	optim.current_yaw = car_pos(2);
	optim.delta_t = delta_t;
	pubOptimize.publish(optim);
	/*Data for analysis with python */

	// calculate measurement updates based on odomoter
	odom_yaw_measured = odom_yaw_measured + odom_delta_yaw_measured;
	odom_yaw_measured = angleDifference(odom_yaw_measured);
	odom_x_measured   = odom_x_measured + lin_vel_measured * cos(odom_yaw_measured) * delta_t; 
	odom_y_measured   = odom_y_measured + lin_vel_measured * sin(odom_yaw_measured) * delta_t;

	imu_yaw_measured = imu_yaw_measured + imu_delta_yaw_measured; 
	imu_yaw_measured = angleDifference(imu_yaw_measured);
	imu_x_measured   = imu_x_measured + lin_vel_measured * cos(imu_yaw_measured) * delta_t;
	imu_y_measured   = imu_y_measured + lin_vel_measured * sin(imu_yaw_measured) * delta_t;

	Eigen::VectorXd zk(6);
	zk << imu_x_measured, imu_y_measured, imu_yaw_measured, odom_x_measured, odom_y_measured, odom_yaw_measured;

	// Kalman Filtering starts here;
	ekf_bar(2) = ekf_state(2) + u_opt(1) * delta_t; 
	ekf_bar(2) = angleDifference(ekf_bar(2));
	ekf_bar(0) = ekf_state(0) + u_opt(0) * delta_t * cos(ekf_state(2));
	ekf_bar(1) = ekf_state(1) + u_opt(0) * delta_t * sin(ekf_state(2));



	Eigen::MatrixXd Ak(3,3);
	Ak << 1.0, 0.0, -u_opt(0)  * std::sin(u_opt(1) * delta_t) * delta_t,
		  0.0, 1.0,  u_opt(0) * std::cos(u_opt(1) * delta_t)  * delta_t, 
		  0.0, 0.0, 1.0;


	Pk = Ak * Pk * Ak.transpose() + Qk;

	Eigen::VectorXd y = zk - Hk * ekf_bar;
	Eigen::MatrixXd Sk = Hk * Pk * Hk.transpose() + Rk;
	Eigen::MatrixXd K = Pk * Hk.transpose() * Sk.inverse();
	ekf_state = ekf_bar + K * y;
	ekf_state(2) = angleDifference(ekf_state(2));
	Pk = Pk - K * Hk * Pk;

}

int main(int argc, char **argv) {
	ros::init(argc, argv, "mpc_controller");
	ros::NodeHandle nh;
	ros::ServiceClient trackclient = nh.serviceClient<custom_msg::track>("track");

	int nx = 3;
	int nu = 2;
	int horizon = 30; 
	int steps = 30;
	double target_velocity = 0.15; // m/s
	double v_max = target_velocity;
	double c = 0.134/2;
	double angvel_max = v_max/c;

	ros::Rate rate(steps);
	
	Eigen::MatrixXd track;

	custom_msg::track trackdata;
	if(trackclient.call(trackdata)) {
		int npoints = trackdata.response.n;
		Eigen::MatrixXd t(3,npoints);
		for(int i=0 ; i < npoints; i++) {
			t(0,i) = trackdata.response.x[i];
			t(1,i) = trackdata.response.y[i];
			t(2,i) = trackdata.response.yaw[i];
		}

		track = t;
		ROS_INFO("track data of %d points successfully received!", npoints);
	} else {
		ROS_ERROR("failed to call service track");
	}

	Eigen::Vector2d umin(-v_max, -angvel_max);
	Eigen::Vector2d umax = -1 * umin;
	Eigen::DiagonalMatrix<double, 3> q;
	q.diagonal() << 100.0,100.0,.1;  //90,90,0.8
	Eigen::DiagonalMatrix<double, 2> r;
	r.diagonal() << 0.1,0.1;

	// initilize and start controller
	MPC_Controller mpc(nh, horizon, q, r, track, target_velocity, umin, umax, nx, nu, c);
	while(ros::ok()) {

		ros::spinOnce();
		rate.sleep();
	}
}
