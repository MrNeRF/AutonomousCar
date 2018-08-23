import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    mpcdata = np.genfromtxt('perfectMPCrun.csv', delimiter=',', dtype=float)
    naivedata = np.genfromtxt('perfectNaiverun.csv', delimiter=',', dtype=float)
 
    track = np.genfromtxt('track.csv', delimiter=',', dtype=float)
    #  1: linear velocity
    #  2: angular velocity
    #  3: ekf state x
    #  4: ekf state y
    #  5: ekf state yaw
    #  6: imu state x
    #  7: imu state y
    #  8: imu state yaw
    #  9: odom state x
    # 10: odom state y
    # 11: odom state yaw
    # 12: ground truth x
    # 13: gournd truth y

    #track 
    trackx = track[:,0]
    tracky = track[:,1]

    #mpc data
    mpcekf_x  = mpcdata[:,3] 
    mpcekf_y  = mpcdata[:,4]
    mpcimu_x  = mpcdata[:,6] 
    mpcimu_y  = mpcdata[:,7]
    mpcodom_x = mpcdata[:,9] 
    mpcodom_y = mpcdata[:,10]
    mpcgt_x    = mpcdata[:,12]
    mpcgt_y    = mpcdata[:,13]
    mpcmodel_x  = mpcdata[:,15]
    mpcmodel_y  = mpcdata[:,16]

    #naive data
    naiveekf_x  = naivedata[:,3] 
    naiveekf_y  = naivedata[:,4]
    naiveimu_x  = naivedata[:,6] 
    naiveimu_y  = naivedata[:,7]
    naiveodom_x = naivedata[:,9] 
    naiveodom_y = naivedata[:,10]
    naivegt_x    = naivedata[:,12]
    naivegt_y    = naivedata[:,13]
    naivemodel_x  = naivedata[:,15]
    naivemodel_y = naivedata[:,16]

    plt.subplot(2,2,1)
    #plt.gca().invert_yaxis();
    #plt.axis([0,10,-.8,.8])
    plt.plot(trackx, -tracky, 'm:' ,linewidth=2, label='track')
    plt.plot(mpcekf_x , -mpcekf_y , 'g:' ,linewidth=2, label='ekf')
    plt.plot(mpcodom_x, -mpcodom_y, 'r--',linewidth=2 ,label='odom')
    plt.plot(mpcgt_x   , -mpcgt_y   , 'k-.',linewidth=2 ,label='truth')
    plt.plot(mpcimu_x , -mpcimu_y , 'b-.',linewidth=2 ,label='imu')
    plt.plot(mpcmodel_x , -mpcmodel_y , 'y-.',linewidth=2 ,label='model')
    plt.title("MPC Data")
    plt.legend(loc='lower left')

    plt.subplot(2,2,2)
   # plt.axis([0,10,-.8,.8])
    plt.plot(trackx, tracky, 'm:' ,linewidth=2, label='track')
    plt.plot(naiveekf_x , -naiveekf_y , 'g:' ,linewidth=2, label='ekf')
    plt.plot(naiveodom_x, -naiveodom_y, 'r--',linewidth=2 ,label='odom')
    plt.plot(naivegt_x   , -naivegt_y   , 'k-.',linewidth=2 ,label='truth')
    plt.plot(naiveimu_x , -naiveimu_y , 'b-.',linewidth=2 ,label='imu')
    plt.plot(naivemodel_x , -naivemodel_y , 'y-.',linewidth=2 ,label='model')
    plt.title("Naive Data")
    plt.legend(loc='lower left')

    plt.subplot(2,2,3)
   # plt.axis([0,10,-.8,.8])
    plt.plot(trackx, -tracky, 'm' ,linewidth=2, label='track')
    plt.plot(naivegt_x, -naivegt_y, 'b' ,linewidth=2, label='naive')
    plt.plot(mpcgt_x, -mpcgt_y, 'k' ,linewidth=2, label='mpc')
    plt.title("MPC vs Naive vs Track Ground Truth")
    plt.legend(loc='lower left')

   # plt.subplot(2,2,3)
   # plt.axis([0,10,-3.2,3.2])
   # plt.plot(xaxis, odom_yaw, 'k-',linewidth=2, label='odom')
   # plt.plot(xaxis, imu_yaw, 'g:',linewidth=2, label='imu')
   # plt.plot(xaxis, ekf_yaw, 'r--',linewidth=2,label='ekf')
   # plt.plot(xaxis, car_yaw, 'b-.',linewidth=2,label='state')
   # plt.title("Yaw Value")
   # plt.legend(loc='lower left')
    plt.show(block = True)


