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

    ekf_state = np.ones((6,1))
    ekf_state[0] = odom_x_measured   = imu_x_measured = init_x
    ekf_state[1] = odom_y_measured   = imu_y_measured = init_y
    ekf_state[2] = odom_yaw_measured = imu_yaw_measured = init_yaw
    ekf_state[3] = 1
    ekf_state[4] = 1
    ekf_state[5] = 1

    rightWheelRadius = 0.033
    leftWheelRadius  = 0.033
    base	     = 0.1338

    rightWheelError = 0.004
    leftWheelError  = 0.004
    baseError	    = 0.03

    Hk = np.zeros((6,6), dtype=np.float64)
    Hk[0,0] = Hk[1,1] = Hk[2,2] = .9
    Hk[0,3] = Hk[1,4] = Hk[2,5] = .9
    Rk = np.zeros((6,6))
    Rk[0,0] = 0.2
    Rk[1,1] = 0.2
    Rk[2,2] = 0.1
    Rk[3,3] = 0.2
    Rk[4,4] = 0.2
    Rk[5,5] = 0.3
    Pk = np.zeros((6,6),dtype=np.float64)
    Qk = np.zeros((6,6),dtype=np.float64)

    odom_x =[]
    imu_x = []
    ekf_x = []
    car_x = []

    odom_y= []
    imu_y = [] 
    ekf_y = []
    car_y = []

    odom_yaw =[]
    imu_yaw =[]
    ekf_yaw = []
    car_yaw = []

    xaxis = []
    steps = data.shape[0] - 1
    steps = 10/steps

    for i in range(start,data.shape[0]):
        odom_delta_yaw_measured = data[i,1]
        imu_delta_yaw_measured  = data[i,2]
        lin_vel_measured        = data[i,3]
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
        ekf_bar = np.zeros((6,1),dtype=np.float64)
        ekf_bar[2] = ekf_state.item(2) + optAngVel * delta_t 
        ekf_bar[2] = angleDifference(ekf_bar.item(2))
        ekf_bar[0] = ekf_state.item(0) + optLinVel * delta_t * np.cos(ekf_state.item(2))
        ekf_bar[1] = ekf_state.item(1) + optLinVel * delta_t * np.sin(ekf_state.item(2))
        ekf_bar[3] = ekf_state.item(3)
        ekf_bar[4] = ekf_state.item(4)
        ekf_bar[5] = ekf_state.item(5)

        dDelta   = (ekf_state.item(3) * rightWheelRadius * deltaAngleRightWheel + ekf_state.item(4) * leftWheelRadius * deltaAngleLeftWheel) / 2.0 * delta_t
        phiDelta = (ekf_state.item(3) * rightWheelRadius * deltaAngleRightWheel - ekf_state.item(4) * leftWheelRadius * deltaAngleLeftWheel ) / (ekf_state.item(5) * base) * delta_t

        phi = ekf_state.item(2) +  angleDifference(phiDelta)
        phi = angleDifference(phi)

        dstar_max = dDelta + (np.abs(rightWheelError * deltaAngleRightWheel) + np.abs(leftWheelError * deltaAngleLeftWheel)) / 2.0
        dstar_min = dDelta - (np.abs(rightWheelError * deltaAngleRightWheel) + np.abs(leftWheelError * deltaAngleLeftWheel)) / 2.0
        dDeltaDelta = (dstar_max - dstar_min) / 2.0
        phistar_max = base / (base - np.sign(phiDelta) * np.abs(baseError)) * (phiDelta + (np.abs(rightWheelError * deltaAngleRightWheel) + np.abs(leftWheelError * deltaAngleLeftWheel)) / base)
        phistar_min = base / (base - np.sign(phiDelta) * np.abs(baseError)) * (phiDelta - (np.abs(rightWheelError * deltaAngleRightWheel) + np.abs(leftWheelError * deltaAngleLeftWheel)) / base)
        phiDeltaDelta = (phistar_max - phistar_min) / 2.0

        Ak = np.array([[1.0, 0.0, -dDelta  * np.sin(phi)],
            [0.0, 1.0, dDelta  * np.cos(phi)], 
            [0.0, 0.0, 1.0]], dtype=np.float64)

        AkAugmen = np.zeros((6,6),dtype=np.float64)
        AkAugmen[:3,:3] = Ak
        AkAugmen[3:6,3:6] = np.eye(3)
        AkAugmen[0,3] = rightWheelRadius * deltaAngleRightWheel * np.cos(phi) * 1.0/2.0 - dDelta * rightWheelRadius * deltaAngleRightWheel * np.sin(phi) * 1.0 / (ekf_state.item(5) * base)
        AkAugmen[0,4] = leftWheelRadius  * deltaAngleLeftWheel  * np.cos(phi) * 1.0/2.0 + dDelta * leftWheelRadius  * deltaAngleLeftWheel  * np.sin(phi) * 1.0 / (ekf_state.item(5) * base)
        AkAugmen[0,5] = dDelta * phiDelta * np.sin(phi) * 1.0/(2.0 * ekf_state.item(5))
        AkAugmen[1,3] = rightWheelRadius * deltaAngleRightWheel * np.sin(phi) * 1.0/2.0 + dDelta * rightWheelRadius * deltaAngleRightWheel * np.cos(phi) * 1.0 / (ekf_state.item(5) * base)
        AkAugmen[1,4] = leftWheelRadius   * deltaAngleLeftWheel   * np.sin(phi)  * 1.0/2.0 - dDelta * leftWheelRadius  * deltaAngleLeftWheel  * np.cos(phi)   * 1.0 / (ekf_state.item(5) * base)
        AkAugmen[1,5] = -1.0 * dDelta * phiDelta * np.cos(phi) * 1.0/(2.0 * ekf_state.item(5))
        AkAugmen[2,3] =        rightWheelRadius * deltaAngleRightWheel /   (ekf_state.item(5) * base)
        AkAugmen[2,4] = -1.0 * leftWheelRadius  * deltaAngleLeftWheel   /  (ekf_state.item(5) * base)
        AkAugmen[2,5] = -1.0 * phiDelta / ekf_state.item(5)

        Gk = np.array([[np.cos(phi), -1.0/2.0 * dDelta * np.sin(phi)],
                       [np.sin(phi),  1.0/2.0 * dDelta * np.cos(phi)],
                       [0.0,  1.0] ],dtype=np.float64)

        Qmax = np.eye(2,dtype=np.float64)
        Qmax[0,0] = dDeltaDelta
        Qmax[1,1] = phiDeltaDelta

        # (3,3)     = (3,2) * (2,2) * (2,3)
        Qk[:3,:3] = np.dot(Gk, np.dot(Qmax,Gk.T))
        Qk[3,3] = 2.10
        Qk[4,4] = 2.10
        Qk[5,5] = 2.2

        # (6,6) = (6,6) * (6,6) * (6,6) 
        Pk = AkAugmen @ Pk @ AkAugmen.T + Qk
        #              (3,6)
        y = zk - Hk @ ekf_bar

        #(3,3)            = (3,6) * (6,6) * (6,3) + (3,3)
        Sk = Hk @ Pk @ Hk.T + Rk
        # (6,3) = (6,6) *  (6,3) * (3,3)
        K = Pk @ Hk.T @ np.linalg.inv(Sk)       
        ekf_state = ekf_bar + K @ y
        ekf_state[2] = angleDifference(ekf_state.item(2))
        Pk = Pk - K @ Hk @ Pk
        
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
        print(ekf_state[3:])

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

