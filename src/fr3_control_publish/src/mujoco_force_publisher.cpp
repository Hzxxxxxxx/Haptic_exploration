// ros2 run fr3_control_publish mujoco_touch_tip_publisher  [model] [rate]

#include <rclcpp/rclcpp.hpp>
#include <fr3_control_publish/msg/touch_tip_wrench.hpp>
#include <mujoco/mujoco.h>
#include <string>
#include <chrono>

using fr3_control_publish::msg::TouchTipWrench;

class MujocoTouchTipPublisher : public rclcpp::Node
{
public:
  MujocoTouchTipPublisher(const std::string &model_path, double hz = 100.0)
  : Node("mujoco_touch_tip_publisher"), rate_hz_(hz)
  {
    /* 1) 加载模型 ---------------------------------------------------- */
    char err[1000] = "Could not load model";
    model_ = mj_loadXML(model_path.c_str(), nullptr, err, sizeof(err));
    if (!model_) {
      RCLCPP_FATAL(get_logger(), "mj_loadXML failed: %s", err);
      rclcpp::shutdown();
      return;
    }
    data_ = mj_makeData(model_);
    if (!data_) {
      RCLCPP_FATAL(get_logger(), "mj_makeData failed");
      mj_deleteModel(model_);
      rclcpp::shutdown();
      return;
    }

    /* 2) 解析索引 ---------------------------------------------------- */
    force_id_  = mj_name2id(model_, mjOBJ_SENSOR, "touch_force");
    torque_id_ = mj_name2id(model_, mjOBJ_SENSOR, "touch_torque");
    site_id_   = mj_name2id(model_, mjOBJ_SITE,   "touch_tip");
    if (site_id_ < 0)
      RCLCPP_WARN(get_logger(), "site \"touch_tip\" 未找到");

    /* 3) Publisher --------------------------------------------------- */
    pub_ = create_publisher<TouchTipWrench>("/touch_tip/wrench", 10);

    /* 4) 定时器 ------------------------------------------------------ */
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0/rate_hz_),
      std::bind(&MujocoTouchTipPublisher::timer_cb, this));

    RCLCPP_INFO(get_logger(), "Model=\"%s\"  %.1f Hz publisher ready",
                model_path.c_str(), rate_hz_);
  }

  ~MujocoTouchTipPublisher() override
  {
    if (data_)  mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

private:
  /* ---------------- 定时回调 ---------------- */
  void timer_cb()
  {
    /* A) step simulation */
    mj_step(model_, data_);

    
    /* C) 读 sensor/force & sensor/torque */
    double sf_x=0,sf_y=0,sf_z=0, st_x=0,st_y=0,st_z=0;
    if (force_id_>=0){
      int adr=model_->sensor_adr[force_id_];
      sf_x=data_->sensordata[adr]; sf_y=data_->sensordata[adr+1]; sf_z=data_->sensordata[adr+2];
    }
    if (torque_id_>=0){
      int adr=model_->sensor_adr[torque_id_];
      st_x=data_->sensordata[adr]; st_y=data_->sensordata[adr+1]; st_z=data_->sensordata[adr+2];
    }
    RCLCPP_INFO(get_logger(),
      "[SENSOR] touch_tip force = [%.6f %.6f %.6f]  torque = [%.6f %.6f %.6f]",
      sf_x,sf_y,sf_z, st_x,st_y,st_z);

    /* D) 读 xfrc_applied (body 级别) —— 这部分将作为消息内容 */
    double xf_x=0,xf_y=0,xf_z=0, xt_x=0,xt_y=0,xt_z=0;
    const char* site_name="unknown";
    if (site_id_>=0){
      site_name=model_->names+model_->name_siteadr[site_id_];
      int bid=model_->site_bodyid[site_id_];
      int base=6*bid;
      xf_x=data_->xfrc_applied[base+0];
      xf_y=data_->xfrc_applied[base+1];
      xf_z=data_->xfrc_applied[base+2];
      xt_x=data_->xfrc_applied[base+3];
      xt_y=data_->xfrc_applied[base+4];
      xt_z=data_->xfrc_applied[base+5];
    }
    

    /* E) 封装并发布 (使用 xfrc_applied) */
    TouchTipWrench msg;
    msg.stamp    = get_clock()->now();
    msg.frame_id = "touch_tip";
    msg.force.x  = xf_x;  msg.force.y  = xf_y;  msg.force.z  = xf_z;
    msg.torque.x = xt_x;  msg.torque.y = xt_y;  msg.torque.z = xt_z;
    pub_->publish(msg);

    RCLCPP_INFO(get_logger(),
      "[END_WRENCH] Published force = [%.6f %.6f %.6f]  torque = [%.6f %.6f %.6f]",
      msg.force.x,msg.force.y,msg.force.z,
      msg.torque.x,msg.torque.y,msg.torque.z);
  }

  /* ---------------- 成员变量 ---------------- */
  mjModel *model_{nullptr};
  mjData  *data_{nullptr};
  int force_id_{-1}, torque_id_{-1}, site_id_{-1};

  double rate_hz_;
  rclcpp::Publisher<TouchTipWrench>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

/* ---------------- main ---------------- */
int main(int argc,char**argv)
{
  rclcpp::init(argc,argv);
  std::string model = "/home/mscrobotics2425laptop16/mujoco_ros_ws/src/mujoco_ros2_control/mujoco_ros2_control_demos/mujoco_models/fr3.xml";
  double hz = 100.0;
  if (argc>=2) model=argv[1];
  if (argc>=3) hz=std::stod(argv[2]);

  rclcpp::spin(std::make_shared<MujocoTouchTipPublisher>(model,hz));
  rclcpp::shutdown();
  return 0;
}
