#include <ros/ros.h>
#include <custom_msg/mpc_control.h>
#include <sensor_msgs/Joy.h>
#include <eigen3/Eigen/Dense>

using namespace std;

const double c = 0.13;

class SS_Controller
{
	public:
		SS_Controller();

	private:
		void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
		ros::NodeHandle nh;

		ros::Publisher car_control;
		ros::Subscriber joy_sub;
};


SS_Controller::SS_Controller() {
		// ROS Messages
		car_control = nh.advertise<custom_msg::mpc_control>("mpc_control",1);
		joy_sub = nh.subscribe<sensor_msgs::Joy>("joy", 1, &SS_Controller::joyCallback, this);
	}

void SS_Controller::joyCallback(const sensor_msgs::Joy::ConstPtr& joy) {


	custom_msg::mpc_control control;
	control.ang_vel = -joy->axes[0] * 1.5; // angular velocity
	control.lin_vel = joy->axes[1]; //forwards, backwards

	car_control.publish(control);

}


int main(int argc, char **argv) {
	ros::init(argc, argv, "SS_Controller");
	SS_Controller robot;

	ros::spin();
}
