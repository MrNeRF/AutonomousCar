#include <ros.h>
#include <imu.h>
#include <Matrix.h>
#include <std_msgs/Int32.h>
#include <custom_msg/mpc_control.h>
#include <custom_msg/sensor_measures.h>
#define enrm 10 
#define in1 4
#define in2 5
#define enlm 9
#define in3 6
#define in4 7
#define rightBackWheelEnc 3
#define leftBackWheelEnc 2
#define rightFrontWheelEnc 18
#define leftFrontWheelEnc 19

volatile int leftBackCounter = 0; 
volatile int rightBackCounter = 0;
volatile int leftFrontCounter = 0;
volatile int rightFrontCounter = 0;
const double c = 0.134/2; // half distance between wheels in meter 0.13469/2

Imu compass;
double delta_t = 0.0;
double imu_old_yaw = 0.0;
double odom_yaw = 0.0;
double lin_vel =0.0;

//distance driven
double distance_lw = 0;
double distance_rw = 0;
double leftWheelRadius = 0.033;
double rightWheelRadius = 0.033;
double ticks = 20.0;
// measured motor speed
double current_speed_lm = 0.0;
double current_speed_rm = 0.0;
// misc
double xref_yaw = 0.0;
double uref_vel = 0.0;
double xpos_estimate = 0.0;
double ypos_estimate = 0.0;
double yaw_estimate = 0.0;
double deltaAngleLeftWheel = 0.0;
double deltaAngleRightWheel = 0.0;

// setpoints for motors
double setpoint_lm = 0;
double setpoint_rm = 0;
double epsilon = 0.05;
double old_setpoint_lm = 0.0;
double old_setpoint_rm = 0.0;
// time variables
unsigned long t_past = 0;
unsigned long t_now = 0;

double dt = 0.0;

// const values for car
const double max_vx = 0.2;
const double max_omega = 3.0;

// milliseconds to sample
int sampletime = 50;

// motor variables
int lm_pwm = 0;
int rm_pwm = 0;

// PID CONTROLLER VARS
double integral_rm = 0.0;
double integral_lm = 0.0;
double last_lmserror = 0.0;
double last_rmserror = 0.0;
double last_lms = 0.0;
double last_rms = 0.0;

const double kp = 40.0;
const double ki = 0.0;
const double kd = 1.0;

/* const double kp = 30.0; */
/* const double ki = 4.0; */
/* const double kd = 1.0; */
// -END- PID CONTROLLER VARS

ros::NodeHandle nh;
custom_msg::sensor_measures measures;
ros::Publisher measure_publisher("sensor_measures", &measures);

static volatile unsigned long leftBackDebounce = 0; //micro seconds
static volatile unsigned long leftFrontDebounce = 0;
static volatile unsigned long rightBackDebounce = 0; //micro seconds
static volatile unsigned long rightFrontDebounce = 0;
static const volatile unsigned long debounceTime = 50;

void leftBackTickCounter(){
	if (digitalRead(leftBackWheelEnc) && (micros() - leftBackDebounce > debounceTime) && digitalRead(leftBackWheelEnc) ) {
		leftBackCounter++;
		leftBackDebounce = micros();
	}

}

void leftFrontTickCounter(){
	if(digitalRead(leftFrontWheelEnc) && (micros() - leftFrontDebounce > debounceTime) && digitalRead(leftFrontWheelEnc)) {
		leftFrontCounter++;
		leftFrontDebounce = micros();
	}
}

void rightBackTickCounter() {
	if (digitalRead(rightBackWheelEnc) && (micros() - rightBackDebounce > debounceTime) && digitalRead(rightBackWheelEnc) ) {
		rightBackCounter++;
		rightBackDebounce = micros();
	}	
}

void rightFrontTickCounter() {
	if(digitalRead(rightFrontWheelEnc) && (micros() - rightFrontDebounce > debounceTime) && digitalRead(rightFrontWheelEnc)) {
		rightFrontCounter++;
		rightFrontDebounce = micros();
	}
}

double sign(double x) {
	if (x < 0)
		return -1.0;
	else
		return 1.0;  
}


void calcMotorSpeed() {
	double leftTicks = 0.0;
	double rightTicks = 0.0;
	noInterrupts();
	// measuredMotor Speed
	
	leftTicks = ((double)leftBackCounter + (double)leftFrontCounter) / 2;
	rightTicks = ((double)rightBackCounter + (double)rightFrontCounter) / 2;
	//if(abs(leftBackCounter - leftFrontCounter) > 1)
	//	leftTicks = (double)min(leftBackCounter, leftFrontCounter);
	//else {
	//}

	//if(abs(rightBackCounter - rightFrontCounter) > 1)
	//	rightTicks = (double)min(rightBackCounter, rightFrontCounter);
	//else {
	//}

	
	
	distance_lw += leftTicks  / ticks * 2.0 * PI * leftWheelRadius;  
	distance_rw += rightTicks / ticks * 2.0 * PI * rightWheelRadius;
	deltaAngleLeftWheel  = leftTicks  * 1.0 / ticks * 2.0 * PI;
	deltaAngleRightWheel = rightTicks * 1.0 / ticks * 2.0 * PI; 
	current_speed_lm	= 2.0 * (leftTicks  / ticks * 2.0 * PI * leftWheelRadius) / dt;
	current_speed_rm	= 2.0 * (rightTicks / ticks * 2.0 * PI * rightWheelRadius) / dt;
	leftBackCounter		= 0;
	rightBackCounter	= 0;
	leftFrontCounter	= 0;
	rightFrontCounter	= 0;
	delta_t = dt;

	interrupts();

	double iccRadius = 0.0;
	lin_vel = (current_speed_rm  + current_speed_lm) / 2.0;

	if (current_speed_rm == current_speed_lm) {
		odom_yaw = 0.0;
	} else {
		iccRadius = c * (current_speed_rm + current_speed_lm) / (-current_speed_rm + current_speed_lm);
		odom_yaw = (-current_speed_rm  + current_speed_lm) / (2.0 * c) * dt;
	}

	if (setpoint_lm < 0)
		current_speed_lm *= -1;
	if (setpoint_rm < 0)
		current_speed_rm *= -1;

	double imu_now = compass.getYaw();
	double imu_delta = imu_now - imu_old_yaw;
	imu_old_yaw = imu_now;

	if (imu_delta > PI)
		imu_delta -= 2 * PI;
	else if (imu_delta < -PI)
		imu_delta += 2 * PI;

	if (odom_yaw > PI)
		odom_yaw -= 2 * PI;
	else if (odom_yaw < -PI)
		odom_yaw += 2 * PI;
	

	measures.lin_vel				= lin_vel;
	measures.imu_yaw_delta 	 		= imu_delta;
	measures.sensor_yaw				= imu_now;
	measures.odom_yaw				= odom_yaw;
	measures.dt						= delta_t;
	measures.leftDistance			= distance_lw;
	measures.rightDistance			= distance_rw;
	measures.deltaAngleRightWheel	= deltaAngleRightWheel;
	measures.deltaAngleLeftWheel	= deltaAngleLeftWheel;
	measures.iccRadius				= iccRadius;
	measure_publisher.publish(&measures);
}


void pid_controller() {

	t_now = millis();
	if(t_now - t_past > sampletime) {
		dt = (t_now - t_past) / (double) 1000;
		t_past = t_now;
		calcMotorSpeed();
		double error_lm = setpoint_lm - current_speed_lm; 
		double error_rm = setpoint_rm - current_speed_rm;

		integral_lm += error_lm * dt;
		integral_rm += error_rm * dt;

		lm_pwm += (int)(error_lm * kp + ki * integral_lm  + (error_lm)/dt * kd);
		rm_pwm += (int)(error_rm * kp + ki * integral_rm  + (error_rm)/dt * kd);


		last_lmserror = error_lm;
		last_rmserror = error_rm;

		if(lm_pwm > 255)
			lm_pwm = 255;
		if(lm_pwm < -255)
			lm_pwm = -255;
		if(rm_pwm > 255)
			rm_pwm = 255;
		if(rm_pwm < -255)
			rm_pwm = -255;
	}
}

void setMotorVals() {

	if (setpoint_lm >= 0){
		digitalWrite(in1, LOW);
		digitalWrite(in2, HIGH);
		analogWrite(enlm, lm_pwm); 
	} else {
		digitalWrite(in1, HIGH);
		digitalWrite(in2, LOW);
		analogWrite(enlm, -lm_pwm); 
	}
	if (setpoint_rm >= 0){
		digitalWrite(in3, LOW);
		digitalWrite(in4, HIGH);
		analogWrite(enrm, rm_pwm); 
	} else {
		digitalWrite(in3, HIGH);
		digitalWrite(in4, LOW);
		analogWrite(enrm, -rm_pwm); 
	} 

	if(sign(setpoint_lm) * setpoint_lm < 0.05)
		analogWrite(enlm, 0); 
	if(sign(setpoint_rm) * setpoint_rm < 0.05)
		analogWrite(enrm, 0); 

}


// get control inputs
void cb_control_cmd(const custom_msg::mpc_control &state) {

	// ensure correctness of control vals	
	double v_x   = state.lin_vel;
	double omega = state.ang_vel; 

	if (omega > max_omega)
		omega = max_omega;
	if (omega < -max_omega)
		omega = -max_omega;
	if (v_x > max_vx)
		v_x = max_vx;
	if (v_x < -max_vx)
		v_x = -max_vx;

	// calc reference velocities
	setpoint_lm  =  (v_x + c * omega); // correctign for skewed steering
	setpoint_rm  =  (v_x - c * omega); // here as well

}

ros::Subscriber<custom_msg::mpc_control> sub("/mpc_control", &cb_control_cmd);

void setup() {
	pinMode(leftBackWheelEnc, INPUT_PULLUP);
	attachInterrupt(digitalPinToInterrupt(leftBackWheelEnc), leftBackTickCounter, CHANGE);
	pinMode(leftFrontWheelEnc, INPUT_PULLUP);
	attachInterrupt(digitalPinToInterrupt(leftFrontWheelEnc), leftFrontTickCounter, CHANGE);
	pinMode(rightBackWheelEnc, INPUT_PULLUP);
	attachInterrupt(digitalPinToInterrupt(rightBackWheelEnc), rightBackTickCounter, CHANGE);
	pinMode(rightFrontWheelEnc, INPUT_PULLUP);
	attachInterrupt(digitalPinToInterrupt(rightFrontWheelEnc), rightFrontTickCounter, CHANGE);
	pinMode(enlm, OUTPUT);
	pinMode(enrm, OUTPUT);
	pinMode(in1, OUTPUT);
	pinMode(in2, OUTPUT);
	pinMode(in3, OUTPUT);
	pinMode(in4, OUTPUT);

	pinMode(SDA, INPUT_PULLUP);
	pinMode(SCL, INPUT_PULLUP);
	nh.initNode();
	nh.subscribe(sub);
	nh.advertise(measure_publisher);
	t_now = millis();
	compass.init();
}

void loop() {

	pid_controller();
	setMotorVals();

	nh.spinOnce();
	//delay(20);	
}
