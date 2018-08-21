#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include "ros/ros.h"
#include <custom_msg/mpc_control.h>
#include <custom_msg/car_position.h>
#include <custom_msg/sensor_measures.h>
#include <custom_msg/track.h>
#include "std_msgs/Float64MultiArray.h"
#include "visualization_msgs/Marker.h"

constexpr double pi() {return std::atan(1) * 4;}

class Naive_Controller {
	public: 
		Naive_Controller(ros::NodeHandle &nh_, Eigen::MatrixXd track_, double target_vel_, double max_angular_vel_) :
			 target_velocity(target_vel_), nh(nh_), track(track_)
		{
			dt = 1.0/30.0;
			max_angular_vel = max_angular_vel_;
			number_refpoints = track.cols();

			positionSubscriber = nh.subscribe("car_position", 10, &Naive_Controller::car_position_cb, this);
			measurementSubscriber = nh.subscribe("sensor_measures", 100, &Naive_Controller::measurement_cb, this);
			pubMPC = nh.advertise<custom_msg::mpc_control>("mpc_control", 1);
			pub_mpcmarker = nh.advertise<visualization_msgs::Marker>("mpc_marker", 10);
		}

	private:
		Eigen::Vector2d u_opt;
		int number_refpoints;
		double target_velocity;
		double dt;
		double max_angular_vel;
		int currentIndex;
		ros::NodeHandle nh;
		ros::Subscriber positionSubscriber, measurementSubscriber;
		ros::Publisher pubMPC, pub_mpcmarker;
		Eigen::MatrixXd track;
		void car_position_cb(const custom_msg::car_position::ConstPtr& p);
		void measurement_cb(const custom_msg::sensor_measures::ConstPtr& measure);
		void calc_mpc(Eigen::Vector3d state, bool sentMarker);
		double angleDifference(double diff);

		int getControl(int index, Eigen::VectorXd state);
		void sendMPCMarker(Eigen::Vector2d nextGoal); 

};

void Naive_Controller::measurement_cb(const custom_msg::sensor_measures::ConstPtr& measure) {

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
	u_opt(1) = 2 * target_velocity * curvature;
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

	Eigen::Vector3d state;
	state << p->xpos, p->ypos, p->yaw;
	currentIndex = getControl(currentIndex, state);

	custom_msg::mpc_control control;
	control.lin_vel	= u_opt(0);
	control.ang_vel	= u_opt(1);

	pubMPC.publish(control);
	
	sendMPCMarker(track.block(0,currentIndex,2,1));
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
