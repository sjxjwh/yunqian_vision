#include "gimbal_csv_logger.hpp"

#include <cmath>
#include <iomanip>

namespace tools
{
namespace
{
constexpr double RAD2DEG = 180.0 / M_PI;
}

bool GimbalCsvLogger::open(const std::string & path)
{
  if (path.empty()) return false;
  if (csv_.is_open()) csv_.close();
  has_t0_ = false;
  t0_sec_ = 0.0;
  csv_.open(path, std::ios::out | std::ios::trunc);
  if (!csv_.is_open()) return false;
  csv_ << "t_sec,sim_ts_ns,frame_seq,new_meas,control,fire,distance_m,bullet_speed,"
          "rx_qw,rx_qx,rx_qy,rx_qz,"
          "rx_yaw_deg,rx_pitch_up_deg,rx_yaw_vel_rad_s,rx_pitch_up_vel_rad_s,"
          "tx_yaw_deg,tx_pitch_up_deg,tx_yaw_vel_rad_s,tx_yaw_acc_rad_s2,"
          "tx_pitch_up_vel_rad_s,tx_pitch_up_acc_rad_s2,"
          "ref_yaw_deg,ref_pitch_up_deg,target_yaw_deg,target_pitch_up_deg,"
          "muzzle_yaw_deg,muzzle_pitch_up_deg,muzzle_yaw_vel_rad_s,"
          "muzzle_pitch_up_vel_rad_s\n";
  return true;
}

bool GimbalCsvLogger::is_open() const { return csv_.is_open(); }

void GimbalCsvLogger::write(const GimbalLogSample & sample)
{
  if (!csv_.is_open()) return;
  if (!has_t0_) {
    t0_sec_ = sample.t_sec;
    has_t0_ = true;
  }
  const double rel_t_sec = std::max(0.0, sample.t_sec - t0_sec_);

  csv_ << std::fixed << std::setprecision(6) << rel_t_sec << ',' << sample.sim_ts_ns << ','
       << sample.frame_seq << ',' << (sample.new_meas ? 1 : 0) << ','
       << (sample.control ? 1 : 0) << ',' << (sample.fire ? 1 : 0) << ',' << sample.distance_m
       << ',' << sample.bullet_speed << ',' << sample.gimbal_q.w() << ',' << sample.gimbal_q.x()
       << ',' << sample.gimbal_q.y() << ',' << sample.gimbal_q.z() << ','
       << sample.gimbal_yaw * RAD2DEG << ',' << -sample.gimbal_pitch * RAD2DEG << ','
       << sample.gimbal_yaw_vel << ',' << -sample.gimbal_pitch_vel << ','
       << sample.cmd_yaw * RAD2DEG << ',' << -sample.cmd_pitch * RAD2DEG << ','
       << sample.cmd_yaw_vel << ',' << sample.cmd_yaw_acc << ',' << -sample.cmd_pitch_vel << ','
       << -sample.cmd_pitch_acc << ',' << sample.ref_yaw * RAD2DEG << ','
       << -sample.ref_pitch * RAD2DEG << ',' << sample.target_yaw * RAD2DEG << ','
       << -sample.target_pitch * RAD2DEG << ',' << sample.muzzle_yaw * RAD2DEG << ','
       << -sample.muzzle_pitch * RAD2DEG << ',' << sample.muzzle_yaw_vel << ','
       << -sample.muzzle_pitch_vel << '\n';
}

}  // namespace tools
