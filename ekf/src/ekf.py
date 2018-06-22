import numpy as np
import matplotlib.pyplot as plt


def angleDifference(angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle

if __name__ == "__main__":

    data = np.genfromtxt('optimize.csv', delimiter=',')

    start = 1
    init_x  = data[start,8];
    init_y  = data[start,9];
    init_yaw= data[start,10];

    ekf_state = np.ones((3,1))
    ekf_state[0] = odom_x_measured   = imu_x_measured   = init_x
    ekf_state[1] = odom_y_measured   = imu_y_measured   = init_y
    ekf_state[2] = odom_yaw_measured = imu_yaw_measured = init_yaw

    Hk = np.zeros((6,3), dtype=np.float64)
    Hk[0,0] = Hk[1,1] = Hk[2,2] = 1
    Hk[3,0] = Hk[4,1] = Hk[5,2] = 1
    Rk = np.zeros((6,6))
    Rk[0,0] = .5 
    Rk[1,1] = .5
    Rk[2,2] = .1
    Rk[3,3] = .4
    Rk[4,4] = .4
    Rk[5,5] = .4
    Pk = np.ones((3,3),dtype=np.float64)
    Pk *=0
    Qk = np.zeros((3,3),dtype=np.float64)
    Qk[0,0] = 0.1
    Qk[1,1] = 0.1
    Qk[2,2] = 100
    Qk[2,0] = Qk[0,2] = 1.0
    Qk[1,0] = Qk[0,1] = 1.0
    Qk[1,2] = Qk[2,1] = 1.

    odom_x =[]
    imu_x = []
    ekf_x = []
    car_x = []

    odom_y =[]
    imu_y = []
    ekf_y = []
    car_y = []

    odom_yaw =[]
    imu_yaw = []
    ekf_yaw = []
    car_yaw = []

    xaxis = []
    steps = data.shape[0] - 1
    steps = 10/steps
    uopt_angle = init_yaw

    for i in range(start,data.shape[0]):
        odom_delta_yaw_measured = data[i,1]
        imu_delta_yaw_measured  = data[i,2]
        lin_vel_measured        = data[i,3] * 1.1
        deltaAngleLeftWheel     = data[i,4]
        deltaAngleRightWheel    = data[i,5]
        optLinVel               = data[i,6] 
        optAngVel               = data[i,7]
        state_x                 = data[i,11]
        state_y                 = data[i,12]
        state_yaw               = data[i,13]
        delta_t                 = data[i,14]

        odom_yaw_measured = odom_yaw_measured + odom_delta_yaw_measured
        odom_yaw_measured = angleDifference(odom_yaw_measured)
        odom_x_measured   = odom_x_measured + lin_vel_measured * np.cos(odom_yaw_measured) * delta_t 
        odom_y_measured   = odom_y_measured + lin_vel_measured * np.sin(odom_yaw_measured) * delta_t

        imu_yaw_measured = imu_yaw_measured + imu_delta_yaw_measured 
        imu_yaw_measured = angleDifference(imu_yaw_measured)
        imu_x_measured   = imu_x_measured + lin_vel_measured * np.cos(imu_yaw_measured) * delta_t
        imu_y_measured   = imu_y_measured + lin_vel_measured * np.sin(imu_yaw_measured) * delta_t

        zk = np.ones((6,1), dtype=np.float64);
        zk = np.array([[imu_x_measured], [imu_y_measured], [imu_yaw_measured], [odom_x_measured],[odom_y_measured], [odom_yaw_measured]])

        # Kalman Filtering starts here
        ekf_bar = np.zeros((3,1),dtype=np.float64)
        ekf_bar[2] = ekf_state.item(2) + optAngVel * delta_t 
        ekf_bar[2] = angleDifference(ekf_bar.item(2))
        ekf_bar[0] = ekf_state.item(0) + optLinVel * delta_t * np.cos(ekf_state.item(2))
        ekf_bar[1] = ekf_state.item(1) + optLinVel * delta_t * np.sin(ekf_state.item(2))

        uopt_angle = uopt_angle + delta_t * optAngVel;
        uopt_angle = angleDifference(uopt_angle)
        Ak = np.array([[1.0, 0.0, -optLinVel  * np.sin(uopt_angle) * delta_t],
            [0.0, 1.0, optLinVel  * np.cos(uopt_angle) * optLinVel], 
            [0.0, 0.0, 1.0]], dtype=np.float64)

        Pk = Ak @ Pk @ Ak.T + Qk

        y = zk - Hk @ ekf_bar

        Sk = Hk @ Pk @ Hk.T + Rk
        # (6,6)
        K = Pk @ Hk.T @ np.linalg.inv(Sk)
        ekf_state = ekf_bar + K @ y
        ekf_state[2] = angleDifference(ekf_state.item(2))
        Pk = Pk - K @ Hk @  Pk
        print(K)
        
        # yaw
        odom_yaw.append(odom_yaw_measured)
        imu_yaw.append(imu_yaw_measured)
        ekf_yaw.append(ekf_state.item(2))
        car_yaw.append(state_yaw)

        # y
        odom_y.append(odom_y_measured)
        imu_y.append(imu_y_measured)
        ekf_y.append(ekf_state.item(1))
        car_y.append(state_y)
        
        # x
        odom_x.append(odom_x_measured)
        imu_x.append(imu_x_measured)
        ekf_x.append(ekf_state.item(0))
        car_x.append(state_x)

        xaxis.append(i * steps)


    plt.subplot(2,2,1)
    plt.plot(xaxis, odom_x, 'k-',linewidth=2, label='odom')
    plt.plot(xaxis, imu_x, 'g:',linewidth=2, label='imu')
    plt.plot(xaxis, ekf_x, 'r--',linewidth=2,label='ekf')
    plt.plot(xaxis, car_x, 'b-.',linewidth=2,label='state')
    plt.title("X Value")
    plt.legend(loc='lower left')

    plt.subplot(2,2,2)
    plt.axis([0,10,-.8,.8])
    plt.plot(xaxis, odom_y, 'k-',linewidth=2, label='odom')
    plt.plot(xaxis, imu_y, 'g:',linewidth=2, label='imu')
    plt.plot(xaxis, ekf_y, 'r--',linewidth=2,label='ekf')
    plt.plot(xaxis, car_y, 'b-.',linewidth=2,label='state')
    plt.title("Y Value")
    plt.legend(loc='lower left')

    plt.subplot(2,2,3)
    plt.axis([0,10,-3.2,3.2])
    plt.plot(xaxis, odom_yaw, 'k-',linewidth=2, label='odom')
    plt.plot(xaxis, imu_yaw, 'g:',linewidth=2, label='imu')
    plt.plot(xaxis, ekf_yaw, 'r--',linewidth=2,label='ekf')
    plt.plot(xaxis, car_yaw, 'b-.',linewidth=2,label='state')
    plt.title("Yaw Value")
    plt.legend(loc='lower left')
    plt.show(block = True)


