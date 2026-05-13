#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
const double PI=3.1415926;
class Quaternion{
public:
    double w,x,y,z;
    Quaternion(double w_=1.0,double x_=0.0,double y_=0.0,double z_=0.0)
        :w(w_),x(x_),y(y_),z(z_){}
    ~Quaternion(){}
    void normalize(){
        double norm=sqrt(w*w+x*x+y*y+z*z);
        if(norm>0){
            w/=norm;
            x/=norm;
            y/=norm;
            z/=norm;
        }    
    }
    Quaternion conjugate() const{
        return Quaternion(w,-x,-y,-z);
    }
    Quaternion operator*(const Quaternion& q)const{
        return Quaternion(w*q.w-x*q.x-y*q.y-z*q.z,
            w*q.x+x*q.w+y*q.z-z*q.y,
            w*q.y-x*q.z+y*q.w+z*q.x,
            w*q.z+x*q.y-y*q.x+z*q.w);
    }
    static Quaternion fromEuler(double roll,double pitch,double yaw){
        double cr=cos(roll*0.5);
        double sr=sin(roll*0.5);
        double cp=cos(pitch*0.5);
        double sp=sin(pitch*0.5);
        double cy=cos(yaw*0.5);
        double sy=sin(yaw*0.5);
        Quaternion q;
        q.w=cr*cp*cy+sr*sp*sy;
        q.x=sr*cp*cy-cr*sp*sy;
        q.y=cr*sp*cy+sr*cp*sy;
        q.z=cr*cp*sy-sr*sp*cy;
        return q;
    }
    void toEuler(double& roll,double& pitch,double& yaw)const{
        double sinr_cosp=2.0*(w*x+y*z);
        double cosr_cosp=1.0-2.0*(x*x+y*y);
        roll=atan2(sinr_cosp,cosr_cosp);
        double sinp=2.0*(w*y-z*x);
        if(fabs(sinp)>=1.0){
            pitch=copysign(PI/2.0,sinp);
        }
        else{
            pitch=asin(sinp);
        }
        double siny_cosp=2.0*(w*z+x*y);
        double cosy_cosp=1.0-2.0*(y*y+z*z);
        yaw=atan2(siny_cosp,cosy_cosp);
    }
    void print() const{
        std::cout<<std::fixed<<std::setprecision(2);
        std::cout<<w<<','<<x<<','<<y<<','<<z;
    }
};
class Pose{
public:
    double x,y,z;
    double roll,pitch,yaw; 
    Pose(double x_=0.0,double y_=0.0,double z_=0.0,
            double roll_=0.0,double pitch_=0.0,double yaw_=0.0)
            : x(x_),y(y_),z(z_),roll(roll_),pitch(pitch_),yaw(yaw_){}
    Quaternion getQuaternion() const{
        return Quaternion::fromEuler(roll,pitch,yaw);
    }
    void setFromQuaternion(const Quaternion& q){
        q.toEuler(roll,pitch,yaw);
    }        
    void transformPoint(double local_x,double local_y,double local_z,
                            double& world_x,double& world_y,double& world_z)const{
        Quaternion q=getQuaternion();
        Quaternion point_q(0,local_x,local_y,local_z);
        Quaternion rotate_q=q*point_q*q.conjugate();
        world_x=rotate_q.x+x;
        world_y=rotate_q.y+y;
        world_z=rotate_q.z+z;
    }
    void print() const{
        std::cout<<std::fixed<<std::setprecision(2);
        std::cout<<x<<','<<y<<','<<z<<'\n'<<roll<<','<<pitch<<','<<yaw;
    }
};
class CoordinateTransformer{
public:
    static Pose transformPose(const Pose& pose_in_A,const Pose& transform_A_to_B){
        Pose result;
        Quaternion q_rot=transform_A_to_B.getQuaternion();
        Quaternion point_A(0,pose_in_A.x,pose_in_A.y,pose_in_A.z);
        Quaternion point_rotated=q_rot*point_A*q_rot.conjugate();
        result.x=point_rotated.x+transform_A_to_B.x;
        result.y=point_rotated.y+transform_A_to_B.y;
        result.z=point_rotated.z+transform_A_to_B.z;
        Quaternion q_A=pose_in_A.getQuaternion();
        Quaternion q_result=q_A*q_rot;
        q_result.normalize();
        result.setFromQuaternion(q_result);
        return result;
    }
};
int main(){
    double x,y,z,roll,pitch,yaw;
    std::string temp1,temp2,target_frame;
    std::cin>>x>>y>>z>>roll>>pitch>>yaw;
    std::cin>>temp1>>temp2>>target_frame;
    Pose gimbal_to_camera(2.0,0.0,0.0,0.0,0.0,0.0);
    Pose odom_to_gimbal(0.0,0.0,0.0,-0.1,-0.1,-0.1);
    Pose camera_to_gimbal;
    camera_to_gimbal.x=gimbal_to_camera.x;
    camera_to_gimbal.y=gimbal_to_camera.y;
    camera_to_gimbal.z=gimbal_to_camera.z;
    camera_to_gimbal.setFromQuaternion(gimbal_to_camera.getQuaternion());
    Pose gimbal_to_odom;
    gimbal_to_odom.x=odom_to_gimbal.x;
    gimbal_to_odom.y=odom_to_gimbal.y;
    gimbal_to_odom.z=odom_to_gimbal.z;
    gimbal_to_odom.setFromQuaternion(odom_to_gimbal.getQuaternion());
    Pose pose_in_camera(x,y,z,roll,pitch,yaw);
    Pose pose_in_gimbal=CoordinateTransformer::transformPose(pose_in_camera,camera_to_gimbal);
    Pose pose_in_odom=CoordinateTransformer::transformPose(pose_in_gimbal,gimbal_to_odom);
    pose_in_odom.print();
    return 0;
}
