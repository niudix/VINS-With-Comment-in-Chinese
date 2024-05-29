#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

//mutex就是线程锁，用来进行线程同步，防止不同线程同时对某一数据进行处理
//区分一下，m_buf 是线程锁。imu_buff 是存内容的buffer
std::mutex m_buf;
//m_buf 在读写imu_buff 和 feature_buf 时使用
std::mutex m_state; 
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//每接收到一帧IMU数据后要做的事情
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    //trcik可以借鉴
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //直接用IMU的数值计算当前的位置（纯IMU的odom）
    // R*acc-g (世界系统下去除重力的加速度值)
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    //gyr_0 上一帧陀螺仪的值 angular_velocity这一帧陀螺仪的值 求平均
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    //陀螺仪的值求平均*dt 得到变化角度，转化为四元数右乘上一帧的四元数 ，得到当前姿态的四元数
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
    //当前加速度的值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

//视觉和imu进行对齐
//这里涉及到一些buffer的操作值的学习
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //什么也没收到，直接返回空的一个vector
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //imu的最后一个时间戳小于等于图像的第一个时间戳（这样就无法对齐）。要等待图像的数据，让图像数据中间充满imu数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        //imu最早的时间戳大于等于图像的第一个时间戳（第一帧图像数据之前的imu数据）。没有意义，直接pop扔掉
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        //现在就可以保证两个图像帧之间一定有imu数据
        //取出队列中第一帧图像，然后把第一帧图像删掉
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        //把imu时间戳在图像之前的所有数据全部取出来，存储到IMUs
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//IMU的回调函数
/*
IMU的消息存进buffer以供后续处理
利用IMU实现一个简单里程计（高频里程计）

*/
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();
    
    //两个线程都要用到这个buffer，一个要往buffer里存数值，另一个要从buffer中取出数值。用线程锁保证存和取不会同时发生
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    //通过条件变量让有数据再取
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        //用了lockguard，lockguard在其生命周期里有效，超出生命周期会被析构。lg是lockguard的名字，对m_state这个全局互斥量加锁
        //m_state lock的是imu message
        std::lock_guard<std::mutex> lg(m_state);
        //IMU数据拿到后要进行的操作
        //通过当前的位姿和IMU的数值，解出下一时刻imu的值直接pub出去
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        //这里就直接pub出去了，实现了一个高频的里程计（纯IMU里程计）
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

/*
* 收集到前端光流结果后的回调函数
* 单纯将前端光流pub出来的结果存到一个buffer里面
*/
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
       
        /*
        * 进行阻塞，等待 getMeasurements()).size() != 0时，就继续下一步操作
        * 进入等待的时候lk就会自动解锁
        * lk是一个unique_lock,管理了互斥量m_buf，从而避免m_buf的死锁。m_buf 在读写imu_buff 和 feature_buf 时使用
        * 简单来说，process这个线程会停在这里等待条件满足再继续执行，同时在此时会自动释放lk锁 条件满足可以认为是measurements获取到了值
        */
        //return (measurements = getMeasurements()).size() != 0;是传递给wait的条件函数。这个表达式尝试从getMeasurements函数获取测量数据，并检查这些数据的大小是否不为零。
        //如果数据的大小不为零，lambda表达式返回true，使得wait函数停止等待，并继续执行。如果数据大小为零，则返回false，wait函数会继续等待条件变量被通知（即某个其他线程调用notify_one或notify_all）。
        con.wait(lk, [&]
                 {
                    //将imu数据和图像数据作对齐，然后一起存到measurment里面
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        //用measurment 把imu的数值和图像传入进行处理
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            //measurement 两个元素，第一个是一帧IMU，第二个是对应图像的特征点（应该是很多IMU，但图像特征点只有那一个）
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                //是不是最后一帧imu的数据（如果和图片时间戳相等那就是最后一帧）
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    //当前imu时间和上一帧imu时间的差值
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //又把IMU的数据提取出来，进行处理
                    //进行IMU预积分：
                    // 预积分就是利用公式计算了纯IMU下的一些数据（速度，加速度，姿态更新），并且更新了误差矩阵
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                //对最后一帧进行插值
                else
                {
                    //img_t 图像的时间戳，current_time上一帧IMU的时间戳，t IMU的时间戳
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    //求出一个比例，利用这个比例进行一个插值
                    //时间戳就是插值算出的时间戳，值也是插值的值
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            // 回环检测
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            //构建了一个map，把特征点的id和特征点的位置速度，存储到map
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            //ProcessImage的作用：
            //调用processImage ，传进去的有刚刚构建的map，还有时间戳
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    //读取参数
    readParameters(n);
    //设置参数
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    //注册publisher
    registerPub(n);

    //订阅IMU
    //IMU的回调函数两个作用：
    //1. 把数据存储到buffer里面供后面使用
    //2. 提供一个更高频的纯IMU的odom
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    //订阅前端光流的结果
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    //restar
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);


    //开了一个核心线程（这就是主的处理程序）
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
