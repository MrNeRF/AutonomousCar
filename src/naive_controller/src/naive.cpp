#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include "ros/ros.h"
#include <custom_msg/mpc_control.h>
#include <custom_msg/car_position.h>
#include <custom_msg/sensor_measures.h>
#include <custom_msg/track.h>
#include "std_msgs/Float64MultiArray.h"
#include "visualization_msgs/Marker.h"

constexpr double pi() {return std::atan(1) * 4;}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

class Naive_Controller {
	public: 
		Naive_Controller(ros::NodeHandle &nh_, Eigen::MatrixXd track_, double target_vel_, double max_angular_vel_) :
			 target_velocity(target_vel_), nh(nh_), track(track_), ekf_state(3), ekf_bar(3){
			dt = 1.0/30.0;
			max_angular_vel = max_angular_vel_;
			number_refpoints = track.cols();
			car_pos << 0,0,0;
			currentIndex = 0;

			deltaAngleLeftWheel = 0.0;
			deltaAngleRightWheel = 0.0;
			done = false;
			stopcounter = 0;
			
			// Kalman filter init
			firstrun = true;
			time_begin = std::chrono::high_resolution_clock::now();

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


			positionSubscriber = nh.subscribe("car_position", 10, &Naive_Controller::car_position_cb, this);
			measurementSubscriber = nh.subscribe("sensor_measures", 100, &Naive_Controller::measurement_cb, this);
			pubMPC = nh.advertise<custom_msg::mpc_control>("mpc_control", 1);
			pub_mpcmarker = nh.advertise<visualization_msgs::Marker>("mpc_marker", 10);
		}

	private:
		bool firstrun;
		Eigen::Vector2d u_opt;
		int number_refpoints;
		bool done;
		int stopcounter;
		double target_velocity;
		double dt;
		double max_angular_vel;
		int currentIndex, kalman_index;
		std::chrono::high_resolution_clock::time_point time_begin;
		ros::NodeHandle nh;
		ros::Subscriber positionSubscriber, measurementSubscriber;
		ros::Publisher pubMPC, pub_mpcmarker;
		Eigen::MatrixXd track;
		void car_position_cb(const custom_msg::car_position::ConstPtr& p);
		void measurement_cb(const custom_msg::sensor_measures::ConstPtr& measure);
		void calc_mpc(Eigen::Vector3d state, bool sentMarker);
		Eigen::Vector3d car_pos;
		Eigen::Vector3d init_state;
		double angleDifference(double diff);

		double deltaAngleLeftWheel; 
		double deltaAngleRightWheel;

		int getControl(int index, Eigen::VectorXd state);
		void sendMPCMarker(Eigen::Vector2d nextGoal); 
		//
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

void Naive_Controller::extended_KF(double delta_t) {


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

void Naive_Controller::measurement_cb(const custom_msg::sensor_measures::ConstPtr& measure) {
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


int Naive_Controller::getControl(int index, Eigen::VectorXd state) {
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

	u_opt(0) = target_velocity;
	u_opt(1) = target_velocity * curvature;
	return index % number_refpoints;
}

double Naive_Controller::angleDifference(double diff) {
	if (diff > pi())
		diff -= 2 * pi();
	else if (diff < -pi())
		diff += 2 * pi();
	return diff;
}


void Naive_Controller::car_position_cb(const custom_msg::car_position::ConstPtr& p) {

	auto now = std::chrono::high_resolution_clock::now();
	dt  = std::chrono::duration<double, std::milli>(now - time_begin).count();
	time_begin = std::chrono::high_resolution_clock::now();
	dt = dt / 1000.0;

	Eigen::Vector3d state;
	state << p->xpos, p->ypos, p->yaw;
	currentIndex = getControl(currentIndex, state);
	car_pos = state;
	if (stopcounter < 2000)
		stopcounter  += sgn(currentIndex - stopcounter) * (currentIndex - stopcounter);

	if(firstrun) {
		ekf_state(0) = imu_x_measured   = model(0) = odom_x_measured   = init_state(0) = state(0);
		ekf_state(1) = imu_y_measured   = model(1) = odom_y_measured   = init_state(1) = state(1);
		ekf_state(2) = imu_yaw_measured = model(2) = odom_yaw_measured = init_state(2) = state(2);
		firstrun = false;
	}


	model(0) = model(0) + u_opt(0) * cos(model(2) + dt * u_opt(1)) * dt;
	model(1) = model(1) + u_opt(0) * sin(model(2) + dt * u_opt(1)) * dt;
	model(2) = model(2) + u_opt(1) * dt;

	if (model(2) > pi())
		model(2) -= 2 * pi();
	else if (model(2) < -pi())
		model(2) += 2 * pi();

	custom_msg::mpc_control control;
	if (stopcounter > 2000 && state(0) > 0 && state(1) > 0) {
		done = true;
	}
	else{
		control.lin_vel	= u_opt(0);
		control.ang_vel	= u_opt(1);
	}
		
	if(done){
		control.lin_vel	= 0;
		control.ang_vel	= 0;
		pubMPC.publish(control);
		ros::shutdown();

	}
	
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
	control.optlinvel	= u_opt(0);
	control.optangvel   = u_opt(1);

	pubMPC.publish(control);
}

void Naive_Controller::sendMPCMarker(Eigen::Vector2d nextGoal) {
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
	g.x = nextGoal(0);
	g.y = nextGoal(1);
	points.points.push_back(g);

	points.action = visualization_msgs::Marker::DELETEALL;
	pub_mpcmarker.publish(points);
	points.action = visualization_msgs::Marker::ADD;
	pub_mpcmarker.publish(points);
}



int main(int argc, char **argv) {
	ros::init(argc, argv, "mpc_controller");
	ros::NodeHandle nh;
	ros::ServiceClient trackClient = nh.serviceClient<custom_msg::track>("track");

	int steps = 30;
	double target_velocity = 0.15; // m/s
	double max_angular_vel = 0.9;

	ros::Rate rate(steps);
	
	Eigen::MatrixXd track;

	custom_msg::track trackData;
	if(trackClient.call(trackData)) {
		int npoints = trackData.response.n;
		Eigen::MatrixXd t(3,npoints);
		for(int i=0 ; i < npoints; i++) {
			t(0,i) = trackData.response.x[i];
			t(1,i) = trackData.response.y[i];
			t(2,i) = trackData.response.yaw[i];
		}
		track = t;
		ROS_INFO("Track data of %d points successfully received!", npoints);
	} else {
		ROS_ERROR("Failed to call service track");
	}


	// initilize and start controller
	Naive_Controller naive(nh, track, target_velocity, max_angular_vel);
	while(ros::ok()) {

		ros::spinOnce();
		rate.sleep();
	}
}
