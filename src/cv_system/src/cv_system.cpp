#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <eigen3/Eigen/Dense>
#include "ros/ros.h"
#include <custom_msg/mpc_control.h>
#include <custom_msg/car_position.h>
#include <custom_msg/track.h>
#include "visualization_msgs/Marker.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <cassert>
#include <stdio.h>

using namespace std;
using namespace cv;

constexpr double pi() {return atan(1) * 4;}

Mat cameraMatrix, distCoeff;
Mat rotationMatrix, tvec;
Mat invRotMatrix, invCamMatrix;
vector<Point2f> mpc_data;
Eigen::MatrixXd mpc_track;
vector<Point> pnts;
int img_height = 720;
int img_width = 1280;
double fps = 30.0;
Point2f coord_bias(img_width/2, img_height/2);

Point2d screenToWorld(Mat &uvPoint) {
	Mat tempMat = invRotMatrix * invCamMatrix * uvPoint;
	Mat tempMat2 = invRotMatrix *tvec;	

	double s = 0. + tempMat2.at<double> (2,0);
	s /= tempMat.at<double>(2,0);
	Mat res =  invRotMatrix * (s * invCamMatrix * uvPoint - tvec);
	return Point2d(res.at<double>(0,0) /100, res.at<double>(1,0) /100);
}

Point2d worldToScreen(Mat xyz) {

	Mat res =  cameraMatrix * (rotationMatrix * xyz + tvec);
	double s = res.at<double>(2,0);
	return Point2f(res.at<double>(0,0) / s, res.at<double>(1,0) / s);
}

void createCassiniTrack() {
	double dt = 1.0/fps;

	int len = 700;
	Eigen::ArrayXXd tmptrack(3, len);

	double a = 0.98;
	double b = 1.0;
	tmptrack.row(0) = Eigen::ArrayXd::LinSpaced(len,0,2 * pi());
	tmptrack.row(1) = Eigen::ArrayXd::LinSpaced(len,0,2 * pi());
	for (int i = 0; i < len; i++) {
		double r = sqrt(pow(a,2) * cos(2*tmptrack(0,i)) + sqrt(pow(b,4) - pow(a,4) * pow(sin(2 * tmptrack(0,i)),2)));
		tmptrack(0,i) = r * cos(tmptrack(0,i)) * 0.6;
		tmptrack(1,i) = r * sin(tmptrack(1,i)) * 0.6;

	}

	// track from 0 to 2pi ends in wrong optimization values
	len = len - 1;
	Eigen::ArrayXXd track = tmptrack.block(0,0,3,len);

	double dx = (track(0,0) - track(0,len - 1)) /dt;
	double dy = (track(1,0) - track(1,len - 1)) /dt;
	track(2,0) = std::atan2(dy, dx);
	assert(!std::isinf(track(2,0)));

	Mat WToS = Mat::zeros(3,1,DataType<double>::type);
	WToS.at<double>(0,0) = track(0,0) * 100;
	WToS.at<double>(1,0) = track(1,0) * 100;
	Point2f tmp = worldToScreen(WToS);
	pnts.push_back(tmp + coord_bias);
	
	for(int i=1;i<len;i++) {
		dx = (track(0,i) - track(0,i-1)) /dt;
		dy = (track(1,i) - track(1,i-1)) /dt;
		track(2,i) = std::atan2(dy, dx);
		assert(!std::isinf(track(2,i)));
		WToS.at<double>(0,0) = track(0,i) * 100;
		WToS.at<double>(1,0) = track(1,i) * 100;
		tmp = worldToScreen(WToS);
		pnts.push_back(tmp + coord_bias);
	}

	// save mpc_data in global variable
	mpc_track = track;
}
void createInfinityTrack() {
	double dt = 1.0/fps;

	int len = 600;
	Eigen::ArrayXXd tmptrack(3, len);

	tmptrack.row(0) = Eigen::ArrayXd::LinSpaced(len,0,2 * pi());
	tmptrack.row(1) = Eigen::ArrayXd::LinSpaced(len,0,2 * pi());
	for (int i = 0; i < len; i++) {
		tmptrack(0,i) = 1.3 * sin(tmptrack(0,i))      * 0.45;
		tmptrack(1,i) = 0.8 * sin(2.0 *tmptrack(1,i)) * 0.45;
		std::cout << tmptrack(0,i) << "," << tmptrack(1,i) << std::endl;
	}

	// track from 0 to 2pi ends in wrong optimization values
	len = len - 1;
	Eigen::ArrayXXd track = tmptrack.block(0,0,3,len);

	double dx = (track(0,0) - track(0,len - 1)) /dt;
	double dy = (track(1,0) - track(1,len - 1)) /dt;
	track(2,0) = std::atan2(dy, dx);
	assert(!std::isinf(track(2,0))); // check if something goes numerically wrong

	Mat WToS = Mat::zeros(3,1,DataType<double>::type);
	WToS.at<double>(0,0) = track(0,0) * 100;
	WToS.at<double>(1,0) = track(1,0) * 100;
	Point2f tmp = worldToScreen(WToS);
	pnts.push_back(tmp + coord_bias);
	
	for(int i=1;i<len;i++) {
		dx = (track(0,i) - track(0,i-1)) /dt;
		dy = (track(1,i) - track(1,i-1)) /dt;
		track(2,i) = std::atan2(dy, dx);
		assert(!std::isinf(track(2,i)));
		WToS.at<double>(0,0) = track(0,i) * 100;
		WToS.at<double>(1,0) = track(1,i) * 100;
		tmp = worldToScreen(WToS);
		pnts.push_back(tmp + coord_bias);
	}

	// save mpc_data in global variable
	mpc_track = track;
}

void createCircleTrack(double screen_radius, double ticks) {
	double dt = 1.0/fps;

	Mat SToW =  Mat::ones(3,1,DataType<double>::type);
	Point2f p;
	SToW.at<double>(0,0) = screen_radius;
	SToW.at<double>(1,0) = screen_radius;
	p = screenToWorld(SToW);
	double world_radius = p.x;

	int len = (int)std::round((2 * pi() * world_radius) / ticks);
	Eigen::ArrayXXd tmptrack(3, len);

	tmptrack.row(0) = Eigen::ArrayXd::LinSpaced(len,0,2 * pi());
	tmptrack.row(1) = Eigen::ArrayXd::LinSpaced(len,0,2 * pi());
	tmptrack.row(0) = (tmptrack.row(0).cos())* world_radius;
	tmptrack.row(1) = (tmptrack.row(1).sin())* world_radius;

	// track from 0 to 2pi ends in wrong optimization values
	len = len - 1;
	Eigen::ArrayXXd track = tmptrack.block(0,0,3,len);

	double dx = (track(0,0) - track(0,len - 1)) /dt;
	double dy = (track(1,0) - track(1,len - 1)) /dt;
	track(2,0) = std::atan2(dy, dx);
	assert(!std::isinf(track(2,0)));

	Mat WToS = Mat::zeros(3,1,DataType<double>::type);
	WToS.at<double>(0,0) = track(0,0) * 100;
	WToS.at<double>(1,0) = track(1,0) * 100;
	Point2f tmp = worldToScreen(WToS);
	pnts.push_back(tmp + coord_bias);
	
	for(int i=1;i<len;i++) {
		dx = (track(0,i) - track(0,i-1)) /dt;
		dy = (track(1,i) - track(1,i-1)) /dt;
		track(2,i) = std::atan2(dy, dx);
		assert(!std::isinf(track(2,i)));
		WToS.at<double>(0,0) = track(0,i) * 100;
		WToS.at<double>(1,0) = track(1,i) * 100;
		tmp = worldToScreen(WToS);
		pnts.push_back(tmp + coord_bias);
	}

	// save mpc_data in global variable
	mpc_track = track;
}

// for the moment only a circular track
bool getTrack(custom_msg::track::Request &req, custom_msg::track::Response &res) {

	int len   = mpc_track.cols();
	res.n	  = len;
	for(int i=0; i < len; i++) {
		res.x.push_back(mpc_track(0,i)); 
		res.y.push_back(mpc_track(1,i)); 
		res.yaw.push_back(mpc_track(2,i));
	}

	ROS_INFO("Track created and sent to Controller!");

	return true;
}

//
double imu_x    = 0.0;
double imu_y    = 0.0;
double imu_yaw  = 0.0;
double odom_x    = 0.0;
double odom_y    = 0.0;
double odom_yaw  = 0.0;
Point3d xk_ekf_screen;

//
void car_callback(const custom_msg::mpc_control::ConstPtr mpc) {

	// EKF in screen coordinates
	Mat WToS = Mat::zeros(3,1,DataType<double>::type);
	WToS.at<double>(0,0) = mpc->ekf_x * 100;
	WToS.at<double>(1,0) = mpc->ekf_y * 100;

	Point2f p = worldToScreen(WToS);
	xk_ekf_screen.x = p.x;// + coord_bias.x;
	xk_ekf_screen.y = p.y;// + coord_bias.y;
	xk_ekf_screen.z = mpc->ekf_yaw;

	// IMU measurements in screen coordinates
	WToS.at<double>(0,0) = mpc->imu_x * 100;
	WToS.at<double>(1,0) = mpc->imu_y * 100;
	p			= worldToScreen(WToS);
	imu_x		= p.x ;//+ coord_bias.x;
	imu_y		= p.y ;//+ coord_bias.y;
	imu_yaw     = mpc->imu_yaw;
	// Odometry
	WToS.at<double>(0,0) = mpc->odom_x * 100;
	WToS.at<double>(1,0) = mpc->odom_y * 100;
	p			= worldToScreen(WToS);
	odom_x		= p.x ;//+ coord_bias.x;
	odom_y		= p.y ;//+ coord_bias.y;
	odom_yaw     = mpc->imu_yaw;
}


void mpc_callback(const visualization_msgs::Marker::ConstPtr msg) {
	mpc_data.clear();
	for (auto iter = msg->points.begin(); iter != msg->points.end(); iter++) {
		Mat WToS = Mat::zeros(3,1,DataType<double>::type);
		WToS.at<double>(0,0) = iter->x * 100;
		WToS.at<double>(1,0) = iter->y * 100;
		Point2f p = worldToScreen(WToS);
		mpc_data.push_back(p);
	}
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "cv_system");
	ros::NodeHandle n;
	ros::Publisher car_pos_pub = n.advertise<custom_msg::car_position>("/car_position", 10);
	ros::Rate loop_rate(30);
	ros::Subscriber sub = n.subscribe("mpc_marker", 10, &mpc_callback);
	ros::Subscriber ekf_sub = n.subscribe("mpc_control", 1, &car_callback);
	ros::ServiceServer trackService = n.advertiseService("track", getTrack);


	FileStorage fs("/home/paja/ros/ssnode/src/cv_system/src/out_camera_data.xml", FileStorage::READ);
	if(!fs.isOpened()) {
		cerr << "Failed to read camera calibration data. Please calibrate first and save as \
			out_camera_data.xml" << endl;
		return EXIT_FAILURE;
	}

	fs["Camera_Matrix"] >> cameraMatrix;
	fs["Distortion_Coefficients"] >> distCoeff;
	fs.release();

	FileStorage fs2("/home/paja/ros/ssnode/src/cv_system/src/measurement.xml", FileStorage::READ);
	if(!fs2.isOpened()) {
		cerr << "Failed to read measurement data form measurement.xml" << endl;
		return EXIT_FAILURE;
	}

	fs2["RotationMatrix"] >> rotationMatrix;
	fs2["tvec"] >> tvec;
	fs2.release();
	invCamMatrix = cameraMatrix.inv();
	invRotMatrix = rotationMatrix.inv();

	VideoCapture cap;
	cap.open(0);
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
	cap.set(CV_CAP_PROP_FOURCC, codec);
	cap.set(CV_CAP_PROP_FPS, 30); 
	cap.set(CV_CAP_PROP_FRAME_WIDTH, img_width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, img_height); 
	cap.set(CV_CAP_PROP_AUTOFOCUS, 0);

	namedWindow("window", CV_WINDOW_AUTOSIZE);

	int dilation_type = MORPH_RECT;
	int erosion_type  = MORPH_RECT;
	int erosion_size  = 2;
	int dilation_size = 8;

	Mat diletion_elem = getStructuringElement( dilation_type, Size( 2*dilation_size + 1, 2*dilation_size+1 ), Point( dilation_size, dilation_size ) );
	Mat erosion_elem  = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );
	Mat background;
	cap >> background;
	cvtColor(background, background, COLOR_BGR2GRAY);
	
	// Variables for getting frames;
	time_t time_begin = time(0);
	int framecounter = 0;

	// create Track Data and get Radius for printing circle
	double track_radius = 300;
	double ticks = 0.005;
	//createCircleTrack(track_radius, ticks);
	//createCassiniTrack();
	createInfinityTrack();
	
	// Parameters for recording Video ~ Press r
	bool recording = false;
	VideoWriter video("autonomous_car.avi", codec, 24, Size(img_width, img_height));

	const Point *pts = (const Point*) Mat(pnts).data;
	int npts = Mat(pnts).rows;
	// Position of car varibles
	Point2f COM(0.0, 0.0), heading(0.0, 0.0);
	while (ros::ok())
	{
		framecounter++;
		Mat gray, bkgr_converted, image, result, mergedFrame(background.size(), background.type());
		Mat erosion, diletion;
		cap >> image;
		cvtColor(image, gray, COLOR_BGR2GRAY);
		cvtColor(background, bkgr_converted, CV_GRAY2RGB);

		subtract(gray, background, result);
		GaussianBlur(result, result, Size(11,11), 3.5, 3.5);
		threshold(result, result, 40, 255, THRESH_BINARY);
		//threshold(result, result, 30, 255, THRESH_BINARY);
		//erode(result, erosion, erosion_elem);
		dilate(result, diletion, diletion_elem);


		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		findContours(diletion, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<vector<Point> > contours_poly( contours.size() );
		vector<Point2f> center( contours.size() );
		vector<float> radius( contours.size() );

		for( size_t i = 0; i < contours.size(); i++ )
		{
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			minEnclosingCircle( contours_poly[i], center[i], radius[i] );
		}

		for( size_t i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar( 255, 255,255);
			circle(image, center[i], (int)radius[i], color, 2, 8, 0 );
		}


		if(contours.size() == 2) {
			COM = center[0] + 0.5 * (center[1] - center[0]);
			if(radius[0] >= radius[1]) {
				heading = 0.5 * (center[1] - center[0]);
				COM = center[0] + heading;
			} else {
				heading = 0.5 * (center[0] - center[1]);
				COM = center[1] + heading;
			}
		}
		
		arrowedLine(image, COM, (COM + 2 * heading), Scalar(0,0,0), 5);

		double yaw = std::atan2(heading.y, heading.x);

		// Publish ROS Message
		custom_msg::car_position pos;
		Mat SToW =  Mat::ones(3,1,DataType<double>::type);
		SToW.at<double>(0,0) = COM.x - coord_bias.x;
		SToW.at<double>(1,0) = COM.y - coord_bias.y;
		Point2f p;
		p = screenToWorld(SToW);
		pos.xpos = p.x;
		pos.ypos = p.y;
		pos.yaw  = yaw;
		car_pos_pub.publish(pos);
		ros::spinOnce();


		//cout << "Kalman:       " << xk_ekf_screen.x << ",  " << xk_ekf_screen.y << ",  " << xk_ekf_screen.z << endl;
		//cout << "IMU:		   " << imu_x  << ",  " << imu_y  << ",  " << imu_yaw << endl;
		//cout << "Odometry:	   " << odom_x  << ",  " << odom_y  << ",  " << odom_yaw << endl;

		//cout << "Camera:	   " << COM.x - coord_bias.x << ",  " << COM.y - coord_bias.y << ",   " << yaw << endl;

		time_t now = time(0) - time_begin;
		if (now >= 1) {
			fps = framecounter / now;
			framecounter = 0;
			time_begin = time(0);
		}

		ostringstream fps_string;
		fps_string << "FPS: " << fps << "  COM: " << COM.x - coord_bias.x << ", " << COM.y  - coord_bias.y<< "  YAW:" << yaw;
		putText(image, fps_string.str(), Point(0,img_height - 5), CV_FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, CV_AA, false);
		//circle(gray, Point(img_width/2, img_height/2), track_radius, (0,0,255), 2);
		polylines(gray, &pts, &npts, 1, true, Scalar(0,255,0));
		if(!mpc_data.empty()) {
			circle(gray, Point2f(mpc_data[0].x, mpc_data[0].y) + coord_bias, 10, (0,0,255),3);
			circle(gray, Point2f(mpc_data[1].x, mpc_data[1].y) + coord_bias, 4, (0,0,255),6);
			for(auto i=2; i < mpc_data.size(); ++i) {
				circle(gray, Point2f(mpc_data[i].x, mpc_data[i].y) +  coord_bias, 1, (0,0,255),2);
			}

		}
		//
		Mat first_row(Size(mergedFrame.rows/2, mergedFrame.cols), mergedFrame.type()), second_row;
		circle(image, Point2f(xk_ekf_screen.x + coord_bias.x, xk_ekf_screen.y + coord_bias.y), 5, (0,0,255),2);

		resize(image, image, mergedFrame.size()/2);
		resize(bkgr_converted, bkgr_converted, mergedFrame.size()/2);
		hconcat(image, bkgr_converted, first_row);

		resize(diletion, diletion, mergedFrame.size()/2);
		resize(gray, gray, mergedFrame.size()/2);
		hconcat(diletion, gray, second_row);
		cvtColor(second_row, second_row, CV_GRAY2RGB);
		vconcat(first_row, second_row,  mergedFrame);

		imshow("window", mergedFrame); 

		char k = waitKey(10);

		switch(k) {
			case 27: 
				ros::shutdown();//ESC
				break;
			case 'r':
				if(!recording) {
					cout << "Recording" << endl;
					video = VideoWriter("autonomous_car.avi", codec, 24, Size(img_width, img_height));
					recording = true;
				} else {
					cout << "Stop Recording" << endl;
					recording = false;
				}
				break;
		}


		loop_rate.sleep();
		if (recording) {
			video.write(mergedFrame);
		}
	}
}
